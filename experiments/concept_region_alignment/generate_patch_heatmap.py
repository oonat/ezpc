import os
import argparse
import random
import re
import json

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from utils import (
    set_seed, 
    load_dataset_config, 
    DATASET_CONFIG, 
    load_backbone,
    get_patch_embeddings_rn50,
    get_patch_concept_acts,
    heat_for_display
)
from dataset import RawImageDataset


# ============================================================
# Visualization
# ============================================================
def save_heatmap_grid_2xN(
    imgs,
    pos_heats,
    neg_heats,
    out_path,
    class_name,
    pos_name,
    neg_name,
    cmap="jet",
    alpha=0.45,
    dpi=350,
    vmin=0.0,
    vmax=1.0,
    add_colorbar=True,
    title_fontsize=15.0,
    label_fontsize=13.0,
    cb_tick_fontsize=10.0,
    wspace=0.003,
    hspace=0.01,
):
    n = len(imgs)
    assert n == len(pos_heats) == len(neg_heats) and n > 0
    is_pdf = out_path.lower().endswith(".pdf")
    alpha_use = 0.7 if is_pdf else alpha

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig_w = 3.0 * n
    fig_h = 4.35
    fig, axes = plt.subplots(2, n, figsize=(fig_w, fig_h), constrained_layout=False)
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j in range(n):
        # POS row
        axes[0, j].imshow(imgs[j], rasterized=is_pdf)
        axes[0, j].imshow(
            pos_heats[j],
            cmap=cmap,
            alpha=alpha_use,
            interpolation="bilinear",
            extent=(0, imgs[j].size[0], imgs[j].size[1], 0),
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        axes[0, j].set_axis_off()

        # NEG row
        axes[1, j].imshow(imgs[j], rasterized=is_pdf)
        axes[1, j].imshow(
            neg_heats[j],
            cmap=cmap,
            alpha=alpha_use,
            interpolation="bilinear",
            extent=(0, imgs[j].size[0], imgs[j].size[1], 0),
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        axes[1, j].set_axis_off()

    axes[0, 0].text(
        -0.065, 0.5, f"POS: {pos_name}",
        transform=axes[0, 0].transAxes,
        rotation=90, va="center", ha="right", fontsize=label_fontsize
    )
    axes[1, 0].text(
        -0.065, 0.5, f"NEG: {neg_name}",
        transform=axes[1, 0].transAxes,
        rotation=90, va="center", ha="right", fontsize=label_fontsize
    )
    fig.suptitle(f"{class_name}", fontsize=title_fontsize, y=0.985)

    right = 0.99 if not add_colorbar else 0.955
    plt.subplots_adjust(
        left=0.06,
        right=right,
        top=0.94,
        bottom=0.01,
        wspace=wspace,
        hspace=hspace
    )

    if add_colorbar:
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.962, 0.12, 0.012, 0.76])
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=cb_tick_fontsize, length=2, pad=1)
        cb.outline.set_linewidth(0.6)

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def sanitize(s):
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)


# ===============================================================
# Main
# ===============================================================
def main(args):
    # Load model and A checkpoint
    model, preprocess, _ = load_backbone("RN50", args.device)
    A = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True).float()

    # Load concept names using the shared config
    _, concept_names = load_dataset_config(args.dataset_root, args.dataset)

    # Build dataset and find images for the given class
    raw_ds = RawImageDataset(args.dataset, args.dataset_root, preprocess=preprocess)
    # Match class_key to the class name used in the dataset
    # RawImageDataset stores samples as (ref, class_name_string)
    class_name = args.class_key.replace("_", " ")
    candidates = [(ref, cn) for ref, cn in raw_ds.samples if cn == class_name]
    if not candidates:
        # Try exact match without underscore replacement (e.g., ImageNet wnid keys)
        # For ImageNet, class_key is a wnid like n02051845, look up the human name
        if args.dataset == "ImageNet-100":
            label_map_path = DATASET_CONFIG['ImageNet-100']['classname_source']['path']
            with open(f"{args.dataset_root}/ImageNet-100/{label_map_path}") as f:
                label_map = json.load(f)
            if args.class_key in label_map:
                class_name = label_map[args.class_key].split(",")[0]
                candidates = [(ref, cn) for ref, cn in raw_ds.samples if cn == class_name]

    if not candidates:
        available = sorted(set(cn for _, cn in raw_ds.samples))
        raise FileNotFoundError(
            f"No images found for class_key='{args.class_key}' (resolved to '{class_name}'). "
            f"Available classes ({len(available)}): {available[:20]}..."
        )

    print(f"Found {len(candidates)} images for class '{class_name}'")

    def find_concept_idx(name):
        for i, c in enumerate(concept_names):
            if c.lower() == name.lower():
                return i
        raise ValueError(f"Concept not found: {name}")

    pos_idx = find_concept_idx(args.pos_concept)
    neg_idx = find_concept_idx(args.neg_concept)

    k = min(args.num_images, len(candidates))
    chosen = random.sample(candidates, k=k)

    imgs, pos_heats, neg_heats = [], [], []

    for ref, _ in chosen:
        img = raw_ds.load_image(ref)

        # Get CLIP patch embeddings
        patch_embs, H, W = get_patch_embeddings_rn50(model, preprocess, img, args.device)

        # Get mean-centered patch concept acts
        z_mc = get_patch_concept_acts(patch_embs, A)

        pos_scores = z_mc[:, pos_idx].reshape(H, W)
        neg_scores = z_mc[:, neg_idx].reshape(H, W)

        pos_heat = heat_for_display(pos_scores)
        neg_heat = heat_for_display(neg_scores)

        imgs.append(img)
        pos_heats.append(pos_heat)
        neg_heats.append(neg_heat)

    # Shared colormap scaling
    vmin = 0.0
    vmax = 1.0

    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(
        args.output_path,
        f"{sanitize(class_name)}_POS_{sanitize(args.pos_concept)}_NEG_{sanitize(args.neg_concept)}.pdf",
    )
    save_heatmap_grid_2xN(
        imgs=imgs,
        pos_heats=pos_heats,
        neg_heats=neg_heats,
        out_path=out_path,
        class_name=class_name,
        pos_name=args.pos_concept,
        neg_name=args.neg_concept,
    )
    print("Saved:", out_path)
    print(f"Used shared cmap range: vmin={vmin:.4f}, vmax={vmax:.4f}")
    print(f"Used {len(imgs)} images (2x{len(imgs)} layout).")


if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(1234)

    parser = argparse.ArgumentParser(description="Concept-Region Alignment Heatmap")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained A matrix checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True, 
                        help="Path to the root dataset folder")
    parser.add_argument("--class_key", required=True,
                        help="Class identifier: human-readable class name "
                             "(e.g., 'Indigo Bunting') "
                             "or ImageNet wnid (e.g., 'n02051845')")
    parser.add_argument("--pos_concept", required=True, 
                        help='e.g. "a blue-gray body"')
    parser.add_argument("--neg_concept", required=True, 
                        help='e.g. "a red face"')
    parser.add_argument("--output_path", type=str, default="./out_heatmaps")
    parser.add_argument("--dataset", required=True, choices=["CUB-200-2011", "ImageNet-100", "Places365"])
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")
    args = parser.parse_args()

    main(args)
