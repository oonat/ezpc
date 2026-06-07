import os
import argparse
import numpy as np
from dataclasses import dataclass
import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    DATASET_CHOICES,
    load_backbone, 
    backbone_to_name, 
    load_dataset_config,
    load_test_embeddings,
    get_text_embs
)

@dataclass
class Metrics:
    top1_agree_pct: float
    seen_acc: float
    unseen_acc: float
    H: float

@torch.no_grad()
def eval_split_generalized(loader, all_classname_embs, A, device):
    correct_ezpc = 0
    agree = 0
    total = 0

    for x, y in tqdm.tqdm(loader, leave=False, desc="Evaluating"):
        x, y = x.to(device), y.to(device)

        clip_logits = x @ all_classname_embs.T
        ezpc_logits = x @ A @ A.T @ all_classname_embs.T

        clip_pred = torch.argmax(clip_logits, dim=1)
        ezpc_pred = torch.argmax(ezpc_logits, dim=1)

        correct_ezpc += torch.sum(ezpc_pred == y).item()
        agree += torch.sum(ezpc_pred == clip_pred).item()
        total += x.shape[0]

    ezpc_acc = correct_ezpc / max(total, 1)
    top1_agree_pct = 100.0 * (agree / max(total, 1))
    return ezpc_acc, top1_agree_pct

def main_one_seed(args, seed, all_text_embs, seen_loader, unseen_loader):
    backbone_name = backbone_to_name(args.backbone)
    m_tag = "all" if args.num_concepts < 0 else str(args.num_concepts)

    base = (f"{args.dataset}_backbone_{backbone_name}_weight_{args.lambda_weight}_"
            f"epoch_{args.num_epochs}_lr_{args.lr}_bs_{args.train_batch_size}")

    output_path = os.path.join(args.output_path, f"{base}_m{m_tag}_seed{seed}")

    if not os.path.isdir(output_path):
        raise FileNotFoundError(f"Missing output folder: {output_path}")

    A = torch.load(os.path.join(output_path, "best_A.pth"), map_location=args.device, weights_only=True)

    seen_acc, seen_agree = eval_split_generalized(seen_loader, all_text_embs, A, args.device)
    unseen_acc, unseen_agree = eval_split_generalized(unseen_loader, all_text_embs, A, args.device)

    n_seen = len(seen_loader.dataset)
    n_unseen = len(unseen_loader.dataset)
    top1_agree = (seen_agree * n_seen + unseen_agree * n_unseen) / max(n_seen + n_unseen, 1)

    H = 2 * seen_acc * unseen_acc / max(seen_acc + unseen_acc, 1e-8)

    return Metrics(top1_agree_pct=float(top1_agree), seen_acc=float(seen_acc), unseen_acc=float(unseen_acc), H=float(H))

def mean_std(xs):
    if len(xs) == 0:
        return 0.0, 0.0
    arr = np.asarray(xs, dtype=np.float64)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(xs) > 1 else 0.0
    return mu, sd

def run(args):
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    all_m = []

    print("====================================")
    print(f"DATASET: {args.dataset}")
    print(f"BACKBONE: {args.backbone}")
    print(f"num_concepts: {args.num_concepts}")
    print(f"seeds: {seeds}")
    print("====================================")

    # Pre-load shared resources
    model, _, tokenizer = load_backbone(args.backbone, args.device)
    classnames, _ = load_dataset_config(args.dataset_root, args.dataset)

    _, seen_embs, seen_ids, unseen_embs, unseen_ids = load_test_embeddings(
        args.dataset_root, args.dataset, args.backbone
    )

    seen_loader = DataLoader(TensorDataset(seen_embs, seen_ids), batch_size=args.eval_batch_size, num_workers=4)
    unseen_loader = DataLoader(TensorDataset(unseen_embs, unseen_ids), batch_size=args.eval_batch_size, num_workers=4)

    # Compute text embeddings once (same classnames/model across all seeds)
    all_text_embs = get_text_embs(model, classnames, args.backbone, tokenizer, args.device)

    for seed in seeds:
        met = main_one_seed(args, seed, all_text_embs, seen_loader, unseen_loader)
        all_m.append(met)
        print(f"seed={seed} | Top1Agree={met.top1_agree_pct:.2f} | SeenAcc={met.seen_acc:.4f} | UnseenAcc={met.unseen_acc:.4f} | H={met.H:.4f}")

    top1_mu, top1_sd = mean_std([m.top1_agree_pct for m in all_m])
    seen_mu, seen_sd = mean_std([m.seen_acc for m in all_m])
    unseen_mu, unseen_sd = mean_std([m.unseen_acc for m in all_m])
    H_mu, H_sd = mean_std([m.H for m in all_m])

    print("====================================")
    print(f"Summary (num_concepts={args.num_concepts}) over {len(seeds)} seeds")
    print(f"Top1Agree: {top1_mu:.2f} ± {top1_sd:.2f}")
    print(f"SeenAcc:   {seen_mu:.4f} ± {seen_sd:.4f}")
    print(f"UnseenAcc: {unseen_mu:.4f} ± {unseen_sd:.4f}")
    print(f"H:         {H_mu:.4f} ± {H_sd:.4f}")
    print("====================================")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multi-Seed Concept size Evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                        help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--num_epochs", type=int, default=10000, 
                        help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--lambda_weight", type=float, default=1.0,
                        help="Lambda weight (reconstruction loss coefficient)")
    parser.add_argument("--train_batch_size", type=int, default=1000000)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_concepts", type=int, required=True, 
                        help="Number of concepts")
    parser.add_argument("--seeds", type=str, default="12,123,1234")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")
    
    args = parser.parse_args()
    run(args)