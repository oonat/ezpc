import os
import torch
import argparse
import numpy as np

from utils import backbone_to_name

def split_seen_unseen(dataset_dir, model_name, seed=42, ratio=0.8):
    rng = np.random.default_rng(seed)

    # Load embeddings and labels
    train_embs = torch.load(os.path.join(dataset_dir, "embeddings", f"{model_name}_train_embeddings.pt"), weights_only=True)
    train_labels = torch.load(os.path.join(dataset_dir, "embeddings", "train_ids.pt"), weights_only=True)
    val_embs = torch.load(os.path.join(dataset_dir, "embeddings", f"{model_name}_test_embeddings.pt"), weights_only=True)
    val_labels = torch.load(os.path.join(dataset_dir, "embeddings", "test_ids.pt"), weights_only=True)

    all_classes = torch.unique(train_labels).tolist()

    rng.shuffle(all_classes)

    # Split into seen/unseen
    n_seen = int(len(all_classes) * ratio)
    seen_classes = set(all_classes[:n_seen])
    unseen_classes = set(all_classes[n_seen:])

    print(f"Total classes: {len(all_classes)}")
    print(f"Seen classes: {len(seen_classes)}, Unseen classes: {len(unseen_classes)}")

    # Filter train set
    seen_mask_train = torch.tensor([lbl.item() in seen_classes for lbl in train_labels])
    unseen_mask_train = torch.tensor([lbl.item() in unseen_classes for lbl in train_labels])

    seen_train_embs = train_embs[seen_mask_train]
    seen_train_labels = train_labels[seen_mask_train]

    unseen_train_embs = train_embs[unseen_mask_train]
    unseen_train_labels = train_labels[unseen_mask_train]

    # Filter val set
    seen_mask_val = torch.tensor([lbl.item() in seen_classes for lbl in val_labels])
    unseen_mask_val = torch.tensor([lbl.item() in unseen_classes for lbl in val_labels])

    seen_val_embs = val_embs[seen_mask_val]
    seen_val_labels = val_labels[seen_mask_val]

    unseen_val_embs = val_embs[unseen_mask_val]
    unseen_val_labels = val_labels[unseen_mask_val]

    # Save splits
    outdir = os.path.join(dataset_dir, "embeddings", "splits")
    os.makedirs(outdir, exist_ok=True)

    torch.save(seen_train_embs, os.path.join(outdir, f"{model_name}_seen_train_embs.pt"))
    torch.save(seen_train_labels, os.path.join(outdir, "seen_train_ids.pt"))

    torch.save(unseen_train_embs, os.path.join(outdir, f"{model_name}_unseen_train_embs.pt"))
    torch.save(unseen_train_labels, os.path.join(outdir, "unseen_train_ids.pt"))

    torch.save(seen_val_embs, os.path.join(outdir, f"{model_name}_seen_test_embs.pt"))
    torch.save(seen_val_labels, os.path.join(outdir, "seen_test_ids.pt"))

    torch.save(unseen_val_embs, os.path.join(outdir, f"{model_name}_unseen_test_embs.pt"))
    torch.save(unseen_val_labels, os.path.join(outdir, "unseen_test_ids.pt"))

    # Save the split mapping
    torch.save({
        "seen_classes": list(seen_classes),
        "unseen_classes": list(unseen_classes)
    }, os.path.join(outdir, "class_split.pt"))

    print(f"Saved seen/unseen splits in {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Dataset folder (e.g., ./CIFAR-100, ./CUB-200-2011, ./ImageNet, ./Places365, ./ImageNet-100)")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for class split")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Ratio of seen classes (rest are unseen)")
    args = parser.parse_args()

    model_name = backbone_to_name(args.backbone)
    split_seen_unseen(args.dataset_dir, model_name, args.seed, args.ratio)
