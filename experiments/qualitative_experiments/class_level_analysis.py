import os
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec

from dataset import RawImageDataset
from utils import (
    DATASET_CHOICES,
    set_seed, 
    load_backbone, 
    load_dataset_config, 
    get_text_embs
)


# ===============================================================
# Concept Scoring
# ===============================================================
def calculate_concept_scores(model, tokenizer, backbone, images, classnames, A_matrix, device):
    """Calculates concept activation scores dynamically for both CLIP and SigLIP."""
    # Generate classname embeddings
    text_emb = get_text_embs(model, classnames, backbone, tokenizer, device)
    
    # Generate image embeddings
    img_emb = F.normalize(model.encode_image(images.to(device)), dim=1).float()
    
    # Compute activation scores using the EZPC A_matrix bottleneck
    scores = (img_emb @ A_matrix) * (text_emb @ A_matrix)
    
    return scores / scores.max(dim=1, keepdim=True).values


# ===============================================================
# Visualization
# ===============================================================
def visualize_class_level(class_name, concept_names, pil_images, save_path, image_size=224):
    """
    Generate a class-level visualization:
    - Left: concept names
    - Right: 3x3 grid of tightly packed images
    - Top: class name centered
    - Bounding box with light background
    """
    fig = plt.figure(figsize=(6.8, 4))
    fig.suptitle(f"Class: {class_name}", fontsize=14, fontweight="bold", y=0.97)

    bbox_left, bbox_bottom, bbox_width, bbox_height = 0.03, 0.03, 0.95, 0.83

    # Background box
    ax_box = fig.add_axes([0, 0, 1, 1])
    ax_box.add_patch(
        Rectangle(
            (bbox_left, bbox_bottom),
            bbox_width,
            bbox_height,
            transform=fig.transFigure,
            linewidth=1.8,
            edgecolor="black",
            facecolor="none",
            joinstyle="round",
            zorder=-1,
        )
    )
    ax_box.axis("off")

    # Left: Concept list
    ax_text = fig.add_axes([0.12, 0.05, 0.22, 0.8])
    ax_text.axis("off")

    ax_text.text(
        0.5, 0.9, "Top Concepts",
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="center",
        transform=ax_text.transAxes
    )

    if concept_names:
        n = len(concept_names)
        y_positions = np.linspace(0.8, 0.1, n)
        for y, txt in zip(y_positions, concept_names):
            ax_text.text(
                0.5, y, txt,
                fontsize=11,
                va="center",
                ha="center",
                wrap=False,
                transform=ax_text.transAxes
            )

    # Right: 3x3 image grid
    grid_left, grid_bottom = 0.49, 0.047
    grid_width, grid_height = 0.48, 0.8

    gs = gridspec.GridSpec(
        3, 3,
        left=grid_left,
        right=grid_left + grid_width,
        bottom=grid_bottom,
        top=grid_bottom + grid_height,
        wspace=0.02,
        hspace=0.02
    )

    for i in range(9):
        ax = fig.add_subplot(gs[i])
        if i >= len(pil_images):
            ax.axis("off")
            continue

        img = pil_images[i].resize((image_size, image_size), Image.LANCZOS)
        ax.imshow(img)
        ax.axis("off")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ===============================================================
# Main
# ===============================================================
def main(args):
    # Load backbone dynamically via utils
    model, preprocess, tokenizer = load_backbone(args.backbone, args.device)
    
    # Load A matrix weights onto the correct device
    A_matrix = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)

    # Load concept names using the universal config from utils.py
    _, concept_names = load_dataset_config(args.dataset_root, args.dataset)

    # Load dataset
    dataset = RawImageDataset(args.dataset, args.dataset_root, preprocess)
    
    # Group sample indices by class
    class_to_indices = {}
    for i, (ref, cls) in enumerate(dataset.samples):
        class_to_indices.setdefault(cls, []).append(i)

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Single class mode or random sampling
    if args.class_name:
        if args.class_name not in class_to_indices:
            raise ValueError(f"Class '{args.class_name}' not found in dataset. "
                             f"Available classes: {sorted(class_to_indices.keys())[:10]}...")
        selected_classes = [args.class_name]
    else:
        available_classes = list(class_to_indices.keys())
        selected_classes = random.sample(available_classes, min(args.num_classes, len(available_classes)))

    for cls_name in selected_classes:
        # Pick up to 9 random sample indices for the grid
        sample_indices = random.sample(class_to_indices[cls_name], min(9, len(class_to_indices[cls_name])))

        # Load raw images for visualization and preprocess for the model
        imgs = [dataset.load_image(dataset.samples[i][0]) for i in sample_indices]
        clip_imgs = torch.stack([dataset.preprocess(img) for img in imgs], dim=0)
        classnames = [cls_name] * clip_imgs.size(0)

        # Compute concept activations
        scores = calculate_concept_scores(model, tokenizer, args.backbone, clip_imgs, classnames, A_matrix, args.device)
        
        # Average the scores across the images and get the top-k concepts
        mean_scores = scores.mean(dim=0)
        topk = torch.topk(mean_scores, args.topk)
        top_concepts = [concept_names[i] for i in topk.indices.cpu().numpy()]

        # Visualize and save
        save_file = os.path.join(args.output_path, f"{cls_name.replace(' ', '_')}_class_level.pdf")
        visualize_class_level(cls_name, top_concepts, imgs, save_file)
        
    print(f"Generated {len(selected_classes)} class-level visualizations in '{args.output_path}'.")


if __name__ == "__main__":
    set_seed(1234)

    parser = argparse.ArgumentParser(description="Class-Level Analysis")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                        help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained A matrix checkpoint")
    parser.add_argument("--output_path", type=str, default="./class_level_outputs", 
                        help="Directory to save the visualizations")
    parser.add_argument("--topk", type=int, default=10, 
                        help="Number of top concepts to display")
    parser.add_argument("--num_classes", type=int, default=40, 
                        help="Number of random classes to visualize")
    parser.add_argument("--class_name", type=str, default=None, 
                        help="Generate visualization for a specific class (skips random sampling)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")
    
    args = parser.parse_args()

    with torch.no_grad():
        main(args)