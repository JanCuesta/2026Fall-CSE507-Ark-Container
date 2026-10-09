#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
import habana_frameworks.torch.core as htcore

from dataloader import VinDrCXR, MIMIC, build_transform_classification
from models import build_omni_model
from utils import get_config, cosine_scheduler


def latest_teacher(init_dir):
    files = glob.glob(os.path.join(init_dir, "models", "weights", "epoch_*", "teacher_cycle_*.pth"))
    if not files:
        raise FileNotFoundError(f"No teacher checkpoint found under {init_dir}")
    def cyc(path):
        m = re.search(r"teacher_cycle_(\d+)_", os.path.basename(path))
        return int(m.group(1)) if m else -1
    path = max(files, key=cyc)
    return path, cyc(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def shared_only(state):
    return {k: v for k, v in state.items() if not k.startswith("omni_heads.")}


def load_shared(model, shared):
    own = model.state_dict()
    with torch.no_grad():
        for k, v in shared.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v.to(own[k].device, dtype=own[k].dtype))


def lr_for_round(i, total, warmup, start_lr, peak_lr, end_lr):
    if total <= 1:
        return peak_lr
    if warmup > 1 and i < warmup:
        return start_lr + (peak_lr - start_lr) * (i / float(warmup - 1))
    if i < warmup:
        return peak_lr
    denom = max(1, total - warmup - 1)
    p = min(1.0, (i - warmup) / float(denom))
    return end_lr + 0.5 * (peak_lr - end_lr) * (1.0 + math.cos(math.pi * p))


def local_epoch(student, teacher, loader, optimizer, momentum, device, name):
    student.train(); teacher.eval()
    bce = torch.nn.BCEWithLogitsLoss(); mse = torch.nn.MSELoss()
    w = (momentum - 0.9) * 5.0
    total = cls = cons = 0.0; n_total = 0
    for xs, xt, y in loader:
        xs = xs.float().to(device); xt = xt.float().to(device); y = y.float().to(device)
        with torch.no_grad():
            ft, _ = teacher(xt, 0)
        fs, pred = student(xs, 0)
        lcls = bce(pred, y); lcons = mse(fs, ft); loss = (1.0 - w) * lcls + w * lcons
        optimizer.zero_grad(set_to_none=True); loss.backward(); htcore.mark_step(); optimizer.step(); htcore.mark_step()
        n = xs.size(0); total += loss.item() * n; cls += lcls.item() * n; cons += lcons.item() * n; n_total += n
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_((1.0 - momentum) * ps.detach().data)
    htcore.mark_step()
    out = {"loss": total / max(1,n_total), "cls": cls / max(1,n_total), "cons": cons / max(1,n_total)}
    print(f"{name} loss={out['loss']:.6f} cls={out['cls']:.6f} cons={out['cons']:.6f} m={momentum:.6f}", flush=True)
    return out


@torch.no_grad()
def mean_auc(model, loader, device):
    model.eval(); ys=[]; ps=[]
    for x, _, y in loader:
        x = x.float().to(device); _, logits = model(x, 0); p = torch.sigmoid(logits); htcore.mark_step()
        ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps); aucs=[]
    for c in range(y.shape[1]):
        if len(np.unique(y[:,c])) >= 2:
            aucs.append(roc_auc_score(y[:,c], p[:,c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def maybe_subset(ds, n):
    return ds if not n or len(ds) <= n else Subset(ds, range(n))


def make_model(classes, device):
    a = SimpleNamespace(model_name="swin_base", projector_features=1376, use_mlp=False, pretrained_weights=None)
    return build_omni_model(a, [classes]).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-dir", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--schedule-rounds", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--peak-lr", type=float, default=3e-4); ap.add_argument("--start-lr", type=float, default=1e-4); ap.add_argument("--end-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-rounds", type=int, default=20); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(); os.makedirs(args.output_dir, exist_ok=True); torch.manual_seed(123); np.random.seed(123)
    device = torch.device("hpu"); cfg = get_config("datasets_config.yaml")
    tr = build_transform_classification(normalize="imagenet", crop_size=224, resize=256, mode="valid")
    te = build_transform_classification(normalize="imagenet", crop_size=224, resize=256, mode="test", test_augment=False)

    vindr_sets=[]
    for rid in range(1,18):
        ds = VinDrCXR(images_path=cfg["VinDrCXR"]["data_dir"], file_path=f"dataset/VinDrCXR/annotators/R{rid}.txt", crop_size=224, resize=256, augment=tr)
        vindr_sets.append(maybe_subset(ds, args.batch_size*2 if args.smoke else 0))
    mimic = MIMIC(images_path=cfg["MIMIC"]["data_dir"], file_path=cfg["MIMIC"]["train_list"], crop_size=224, resize=256, augment=tr)
    mimic = maybe_subset(mimic, args.batch_size*8 if args.smoke else 0)
    vtest = VinDrCXR(images_path=cfg["VinDrCXR"]["data_dir"], file_path=cfg["VinDrCXR"]["test_list"], crop_size=224, resize=256, augment=te)
    vtest = maybe_subset(vtest, 512 if args.smoke else 0)
    vloaders=[DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers) for ds in vindr_sets]
    mloader=DataLoader(mimic,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    tloader=DataLoader(vtest,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)

    vs=make_model(6,device); vt=make_model(6,device); ms=make_model(14,device); mt=make_model(14,device)
    teacher_path, init_cycle = latest_teacher(args.init_dir); print(f"Initializing from {teacher_path} (VinDr cycle {init_cycle})", flush=True)
    base=torch.load(teacher_path,map_location="cpu",weights_only=False)
    if isinstance(base,dict) and "teacher" in base: base=base["teacher"]
    elif isinstance(base,dict) and "state_dict" in base: base=base["state_dict"]
    vs.load_state_dict(base,strict=True); vt.load_state_dict(base,strict=True)
    master=shared_only(base); load_shared(ms,master); mt.load_state_dict(ms.state_dict(),strict=True)
    for model in (vt,mt):
        for p in model.parameters(): p.requires_grad=False
        model.eval()

    start_round=0; latest=os.path.join(args.output_dir,"latest.pt")
    if args.resume and os.path.isfile(latest):
        ck=torch.load(latest,map_location="cpu",weights_only=False); start_round=int(ck["round"]); master=ck["master"]
        vs.load_state_dict(ck["vin_student"],strict=True); vt.load_state_dict(ck["vin_teacher"],strict=True); ms.load_state_dict(ck["mim_student"],strict=True); mt.load_state_dict(ck["mim_teacher"],strict=True)
        print(f"Resumed after federated round {start_round}",flush=True)

    vsched=cosine_scheduler(0.9,1.0,args.schedule_rounds,17); msched=cosine_scheduler(0.9,1.0,args.schedule_rounds,1)
    for ri in range(start_round,args.rounds):
        rn=ri+1; lr=lr_for_round(ri,args.schedule_rounds,args.warmup_rounds,args.start_lr,args.peak_lr,args.end_lr)
        print(f"===== FEDERATED ROUND {rn}/{args.schedule_rounds} lr={lr:.8e} =====",flush=True)
        load_shared(vs,master); load_shared(ms,master)
        vo=torch.optim.AdamW(vs.parameters(),lr=lr,weight_decay=0.0); mo=torch.optim.AdamW(ms.parameters(),lr=lr,weight_decay=0.0)
        for rid,loader in enumerate(vloaders,1):
            it=ri*17+(rid-1); local_epoch(vs,vt,loader,vo,float(vsched[it]),device,f"VinDr_R{rid}")
        local_epoch(ms,mt,mloader,mo,float(msched[ri]),device,"MIMIC")
        vstate=vs.state_dict(); mstate=ms.state_dict(); new={}
        for k in master:
            a=vstate[k].detach(); b=mstate[k].detach(); new[k]=(((a+b)*0.5) if torch.is_floating_point(a) else a).cpu().clone()
        master=new; load_shared(vs,master); load_shared(ms,master)
        teacher_auc=mean_auc(vt,tloader,device); master_eval=make_model(6,device); master_eval.load_state_dict(vs.state_dict(),strict=True); load_shared(master_eval,master); master_auc=mean_auc(master_eval,tloader,device); del master_eval
        print(f"ROUND {rn} VinDr teacher_mAUC={teacher_auc:.6f} master_mAUC={master_auc:.6f}",flush=True)
        ck={"round":rn,"init_vindr_cycle":init_cycle,"master":master,"vin_student":cpu_state(vs),"vin_teacher":cpu_state(vt),"mim_student":cpu_state(ms),"mim_teacher":cpu_state(mt),"teacher_mAUC":teacher_auc,"master_mAUC":master_auc,"peak_lr":args.peak_lr}
        rp=os.path.join(args.output_dir,f"round_{rn:04d}.pt"); torch.save(ck,rp); torch.save(ck,latest); print(f"Saved {rp}",flush=True)
    print("Federated run complete.",flush=True)

if __name__ == "__main__":
    main()
