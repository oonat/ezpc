import os
import json
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from utils import (
    DATASET_CHOICES,
    set_seed, 
    load_backbone, 
    backbone_to_name, 
    load_dataset_config, 
    load_test_embeddings,
    get_text_embs
)


def generate_intervention_plots(stats, output_dir, dataset_name, backbone_name):
    os.makedirs(output_dir, exist_ok=True)

    # Top-10 vs Random-10 removal comparison
    top10 = stats.get("topk_logit_drop", {}).get(10, []) or stats.get("topk_logit_drop_k10", [])
    rand10 = stats.get("rand_logit_drop", {}).get(10, []) or stats.get("rand_logit_drop_k10", [])

    if len(top10) > 0 and len(rand10) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(top10, bins=40, alpha=0.7, label="Top-10 Removal")
        plt.hist(rand10, bins=40, alpha=0.7, label="Random-10 Removal")
        plt.xlabel("Logit Drop")
        plt.ylabel("Count")
        plt.legend()
        plt.title(f"Top-10 vs Random-10 Removal ({dataset_name}, {backbone_name})")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"intervention_removal_compare_{dataset_name}_{backbone_name}.pdf")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")
    else:
        print("Missing logit drop values for k=10, skipping removal comparison plot.")

def faithfulness_analysis(A, loader, class_concepts, device, k_list=(1, 3, 5, 10)):
    num_concepts = A.shape[1]

    stats = {
        "topk_logit_drop": {k: [] for k in k_list},
        "topk_logit_drop_mean": 0,
        "rand_logit_drop": {k: [] for k in k_list},
        "rand_logit_drop_mean": 0,
        "topk_flip_count": {k: 0 for k in k_list},
        "rand_flip_count": {k: 0 for k in k_list},
        "num_samples": 0
    }

    for (batch_embs,) in tqdm.tqdm(loader, desc="Faithfulness analysis..."):
        batch_embs = batch_embs.to(device)
        stats["num_samples"] += batch_embs.shape[0]

        with torch.no_grad():
            # Calculate img concept scores
            img_concepts = batch_embs @ A  # (B, num_concepts)

            # Calculate logits
            logits = img_concepts @ class_concepts.T  # (B, C)

            # Get the predicted classes
            pred_classes = torch.argmax(logits, dim=1)

            # Concept scores for predicted class
            class_concepts_pred = class_concepts[pred_classes]
            concept_scores_pred = img_concepts * class_concepts_pred

            # Iterate over the img samples
            for sample_idx in range(img_concepts.shape[0]):
                sample_logits = logits[sample_idx]
                sample_pred_class = pred_classes[sample_idx].item()
                sample_concept_scores = img_concepts[sample_idx]

                rand_perm = torch.randperm(num_concepts, device=device)

                for k in k_list:
                    k_safe = min(k, num_concepts)
                    
                    # Remove the contribution of top-k concepts
                    _, topk_idx = torch.topk(concept_scores_pred[sample_idx], k=k_safe, largest=True)
                    top_class_weights = class_concepts[:, topk_idx]
                    top_concept_scores = sample_concept_scores[topk_idx].unsqueeze(0)
                    delta_all = (top_class_weights * top_concept_scores).sum(dim=1)
                    s_minus = sample_logits - delta_all
                    
                    stats["topk_logit_drop"][k].append(delta_all[sample_pred_class].item())
                    if torch.argmax(s_minus).item() != sample_pred_class:
                        stats["topk_flip_count"][k] += 1

                    # Remove the contribution of random-k concepts
                    rand_idx = rand_perm[:k_safe]
                    rand_concept_weights = class_concepts[:, rand_idx]
                    rand_concept_scores = sample_concept_scores[rand_idx].unsqueeze(0)
                    delta_r = (rand_concept_weights * rand_concept_scores).sum(dim=1)
                    s_minus_r = sample_logits - delta_r

                    stats["rand_logit_drop"][k].append(delta_r[sample_pred_class].item())
                    if torch.argmax(s_minus_r).item() != sample_pred_class:
                        stats["rand_flip_count"][k] += 1

    stats['topk_logit_drop_mean'] = {k: sum(v)/max(len(v), 1) for k, v in stats['topk_logit_drop'].items()}
    stats['rand_logit_drop_mean'] = {k: sum(v)/max(len(v), 1) for k, v in stats['rand_logit_drop'].items()}

    return stats

# ===============================================================
# Main
# ===============================================================
def main(args):
    backbone_name = backbone_to_name(args.backbone)

    # Load backbone
    model, _, tokenizer = load_backbone(args.backbone, args.device)

    # Load Checkpoint
    A = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)

    # Load Dataset Config
    classnames, _ = load_dataset_config(args.dataset_root, args.dataset)
    class_name_embs = get_text_embs(model, classnames, args.backbone, tokenizer, args.device)
    class_concepts = class_name_embs @ A

    # Load test embeddings
    _, seen_embs, _, unseen_embs, _ = load_test_embeddings(
        args.dataset_root, args.dataset, args.backbone
    )
    
    all_embs = torch.cat([seen_embs, unseen_embs], dim=0)

    # Limit to max_samples if requested
    if args.max_samples:
        perm = torch.randperm(len(all_embs))[:args.max_samples]
        all_embs = all_embs[perm]

    dataset = TensorDataset(all_embs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Running Faithfulness Analysis on {args.dataset} ({backbone_name})...")
    stats = faithfulness_analysis(
        A=A,
        loader=loader,
        class_concepts=class_concepts,
        device=args.device,
        k_list=(1, 3, 5, 10)
    )

    # Save stats and plots
    os.makedirs(args.output_path, exist_ok=True)
    out_json = os.path.join(args.output_path, f"faithfulness_{args.dataset}_{backbone_name}.json")

    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved stats to: {out_json}")

    try:
        pdf_path = os.path.join(args.output_path, f"faithfulness_distributions_{args.dataset}_{backbone_name}.pdf")
        labels, dists = [], []

        for k in (1, 3, 5, 10):
            vals = stats["topk_logit_drop"].get(k, [])
            if vals:
                labels.append(f"k={k}")
                dists.append(vals)

        if dists:
            plt.figure(figsize=(10, 6))
            plt.boxplot(dists, labels=labels)
            plt.title("Faithfulness Logit Drop Distributions (Top-k Concept Removal)")
            plt.ylabel("Logit Drop")
            plt.tight_layout()
            plt.savefig(pdf_path)
            plt.close()
            print(f"Saved PDF plot to: {pdf_path}")
        else:
            print("No data available for PDF generation.")
            
    except Exception as e:
        print(f"PDF generation failed: {e}")

    generate_intervention_plots(stats, args.output_path, args.dataset, backbone_name)


if __name__ == "__main__":
    set_seed(1234)

    parser = argparse.ArgumentParser(description="Faithfulness Analysis")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                        help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained A matrix checkpoint")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--batch_size", type=int, default=512, 
                        help="Batch size for dataloader")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Cap the number of samples analyzed")
    parser.add_argument("--output_path", type=str, default="./faithfulness_outputs", 
                        help="Directory to save the outputs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")

    args = parser.parse_args()
    main(args)