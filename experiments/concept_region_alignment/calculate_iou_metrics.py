import os
import glob
import argparse

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from utils import (
    set_seed, 
    load_dataset_config,
    load_backbone,
    get_patch_embeddings_rn50,
    get_patch_concept_acts,
    heat_for_display
)

# ============================================================
# Segmentation alignment
# ============================================================
def seg_path_from_image_path_cub(img_path):
    # /.../CUB-200-2011/images/<class_key>/xxx.jpg
    # -> /.../CUB-200-2011/segmentations/<class_key>/xxx.png
    seg_path = img_path.replace("/images/", "/segmentations/")
    seg_path = os.path.splitext(seg_path)[0] + ".png"
    return seg_path

def get_clip_geom(preprocess):
    return transforms.Compose(preprocess.transforms[:2])

def load_segmask_patchgrid(seg_path, geom, H, W):
    m = Image.open(seg_path).convert("L")
    m = geom(m)
    m = m.resize((W, H), Image.NEAREST)
    m = np.array(m)
    return (m > 0).astype(np.uint8)

# ============================================================
# Metrics
# ============================================================
def pointing_acc(heat, mask):
    if heat.max() <= 0:
        return 0.0
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    return float(mask[y, x] == 1)

def inside_ratio(heat, mask, eps=1e-8):
    heat = np.maximum(heat, 0.0)
    inside = (heat * mask).sum()
    total = heat.sum() + eps
    return float(inside / total)

def iou_topk(heat, mask, k=10.0, eps=1e-8):
    if heat.size == 0 or heat.max() <= 0:
        return 0.0
    thr = np.percentile(heat, 100.0 - k)
    pred = (heat >= thr).astype(np.uint8)
    inter = (pred & mask).sum()
    union = (pred | mask).sum()
    return float(inter / (union + eps))

def mean_std(x):
    x = np.asarray(x, dtype=np.float32)
    return float(x.mean()), float(x.std())

# ============================================================
# Class resolution
# ============================================================
def resolve_cub_class_dir(dataset_root, class_key):
    """Resolve a class_key to the CUB image folder name (e.g. '089.Indigo_Bunting').

    Accepts the same human-readable form as generate_patch_heatmap.py
    (e.g. 'Indigo_Bunting' or 'Indigo Bunting'), as well as the full folder name.
    """
    images_root = os.path.join(dataset_root, "CUB-200-2011", "images")
    key = class_key.replace(" ", "_")

    # Already a valid folder name (e.g. '089.Indigo_Bunting')
    if os.path.isdir(os.path.join(images_root, class_key)):
        return class_key

    # Match by the name after the numeric prefix
    for name in sorted(os.listdir(images_root)):
        if name.split(".", 1)[-1] == key:
            return name

    raise FileNotFoundError(
        f"Could not resolve class_key '{class_key}' to a CUB image folder under {images_root}"
    )

# ============================================================
# Main
# ============================================================
def main(args):
    model, preprocess, _ = load_backbone("RN50", args.device)
    geom = get_clip_geom(preprocess)

    A = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True).float()

    _, concept_names = load_dataset_config(args.dataset_root, "CUB-200-2011")

    class_dir = resolve_cub_class_dir(args.dataset_root, args.class_key)
    class_name = class_dir.split(".", 1)[-1].replace("_", " ")

    candidates = sorted(
        glob.glob(f"{args.dataset_root}/CUB-200-2011/images/{class_dir}/*.jpg")
    )
    if not candidates:
        raise FileNotFoundError(f"No images found for class_key={args.class_key}")

    def find_concept_idx(name):
        for i, c in enumerate(concept_names):
            if c.lower() == name.lower():
                return i
        raise ValueError(f"Concept not found: {name}")

    pos_idx = find_concept_idx(args.pos_concept)
    neg_idx = find_concept_idx(args.neg_concept)

    # Metrics accumulators
    metrics = {
        "pos_point": [],
        "neg_point": [],
        "pos_inside": [],
        "neg_inside": [],
        "pos_iou10": [],
        "neg_iou10": [],
        "pos_iou20": [],
        "neg_iou20": [],
    }

    # Iterate all images in the class
    for p in candidates:
        img = Image.open(p).convert("RGB")

        # Get CLIP patch embeddings
        patch_embs, H, W = get_patch_embeddings_rn50(model, preprocess, img, args.device)

        # Get mean-centered patch concept acts
        z_mc = get_patch_concept_acts(patch_embs, A)

        pos_scores = z_mc[:, pos_idx].reshape(H, W)
        neg_scores = z_mc[:, neg_idx].reshape(H, W)

        pos_heat = heat_for_display(pos_scores)
        neg_heat = heat_for_display(neg_scores)

        seg_path = seg_path_from_image_path_cub(p)
        if not os.path.exists(seg_path):
            continue

        mask = load_segmask_patchgrid(seg_path, geom, H, W)

        # Metrics
        metrics["pos_point"].append(pointing_acc(pos_heat, mask))
        metrics["neg_point"].append(pointing_acc(neg_heat, mask))

        metrics["pos_inside"].append(inside_ratio(pos_heat, mask))
        metrics["neg_inside"].append(inside_ratio(neg_heat, mask))

        metrics["pos_iou10"].append(iou_topk(pos_heat, mask, k=10.0))
        metrics["neg_iou10"].append(iou_topk(neg_heat, mask, k=10.0))
        metrics["pos_iou20"].append(iou_topk(pos_heat, mask, k=20.0))
        metrics["neg_iou20"].append(iou_topk(neg_heat, mask, k=20.0))

    n_used = len(metrics["pos_point"])
    if n_used == 0:
        raise RuntimeError("No images were evaluated (missing masks or empty folder)!")

    # Summary numbers
    pos_point_m, pos_point_s = mean_std(metrics["pos_point"])
    neg_point_m, neg_point_s = mean_std(metrics["neg_point"])

    pos_in_m, pos_in_s = mean_std(metrics["pos_inside"])
    neg_in_m, neg_in_s = mean_std(metrics["neg_inside"])

    pos_i10_m, pos_i10_s = mean_std(metrics["pos_iou10"])
    neg_i10_m, neg_i10_s = mean_std(metrics["neg_iou10"])

    pos_i20_m, pos_i20_s = mean_std(metrics["pos_iou20"])
    neg_i20_m, neg_i20_s = mean_std(metrics["neg_iou20"])

    print("\n=== Quantitative localization results (CUB segmentation) ===")
    print(f"Class: {class_name} ({args.class_key})")
    print(f"POS concept: {args.pos_concept}")
    print(f"NEG concept: {args.neg_concept}")
    print(f"Images evaluated: {n_used} / {len(candidates)}")
    print("")
    print(f"Pointing Acc (POS): {pos_point_m:.3f} ± {pos_point_s:.3f}")
    print(f"Pointing Acc (NEG): {neg_point_m:.3f} ± {neg_point_s:.3f}")
    print(f"Inside Ratio (POS): {pos_in_m:.3f} ± {pos_in_s:.3f}")
    print(f"Inside Ratio (NEG): {neg_in_m:.3f} ± {neg_in_s:.3f}")
    print(f"IoU@10% (POS):      {pos_i10_m:.3f} ± {pos_i10_s:.3f}")
    print(f"IoU@10% (NEG):      {neg_i10_m:.3f} ± {neg_i10_s:.3f}")
    print(f"IoU@20% (POS):      {pos_i20_m:.3f} ± {pos_i20_s:.3f}")
    print(f"IoU@20% (NEG):      {neg_i20_m:.3f} ± {neg_i20_s:.3f}")


if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained A matrix checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--class_key", required=True,
                        help="e.g., 'Indigo Bunting'")
    parser.add_argument("--pos_concept", required=True, 
                        help='e.g. "a blue-gray body"')
    parser.add_argument("--neg_concept", required=True, 
                        help='e.g. "a red face"')
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")

    args = parser.parse_args()
    main(args)
