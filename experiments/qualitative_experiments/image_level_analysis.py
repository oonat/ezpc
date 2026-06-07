import os
import argparse
import random

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import (
    DATASET_CHOICES,
    set_seed, 
    load_backbone, 
    load_dataset_config, 
    get_text_embs
)
from dataset import RawImageDataset


# ===============================================================
# Visualization
# ===============================================================
def visualize_top_concepts(img, class_name, scores, concept_names, save_file):
    topk = torch.topk(scores, 10, largest=True)
    top_indices, top_vals = topk.indices, topk.values
    top_concepts = [concept_names[i] for i in top_indices]

    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2,
        figsize=(10, 4.5),                
        gridspec_kw={'width_ratios': [1.4, 1.2]}  
    )

    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(f"Class: {class_name}", fontsize=16, fontweight="bold", pad=10)

    y_pos = range(len(top_vals))
    top_vals_np = top_vals.cpu().numpy()
    bars = ax_bar.barh(y_pos, top_vals_np, color="#2E8B57", alpha=0.9)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(top_concepts, fontsize=13, fontweight="medium")
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Concept Score", fontsize=13)
    ax_bar.set_title("Most Strongly Contributing Concepts", fontsize=14, fontweight="bold", pad=10)
    ax_bar.tick_params(axis="x", labelsize=10)

    for bar, val in zip(bars, top_vals_np):
        ax_bar.text(
            val - 0.02 * max(top_vals_np),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            ha="right",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    plt.tight_layout(pad=2.0)
    plt.savefig(save_file, bbox_inches="tight", dpi=300)
    plt.close()


# ===============================================================
# Concept Evaluation Logic
# ===============================================================
def interpret(model, tokenizer, preprocess, img, class_name, concept_names, A_matrix, backbone, device, output_path, fname):
    # Image embedding
    image_tensor = preprocess(img).unsqueeze(0).to(device)
    img_embed = F.normalize(model.encode_image(image_tensor), dim=1).float()

    # Text embedding via universal utils function
    text_embed = get_text_embs(model, [class_name], backbone, tokenizer, device).float()

    # Calculate scores across the A matrix bottleneck
    img_scores = (img_embed @ A_matrix).squeeze()
    text_scores = (text_embed @ A_matrix).squeeze()
    concept_scores = img_scores * text_scores

    # Visualizations
    save_file = os.path.join(output_path, fname)
    visualize_top_concepts(img, class_name, concept_scores, concept_names, save_file)

# ===============================================================
# Main
# ===============================================================
def main(args):
    # Load backbone dynamically via utils
    model, preprocess, tokenizer = load_backbone(args.backbone, args.device)
    
    # Load A matrix weights onto the correct device
    A_matrix = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True).float()

    # Load universal configs
    classnames, concept_names = load_dataset_config(args.dataset_root, args.dataset)

    os.makedirs(args.output_path, exist_ok=True)

    # Load dataset
    dataset = RawImageDataset(args.dataset, args.dataset_root, preprocess)

    # Group sample indices by class for easy random sampling
    class_to_indices = {}
    for i, (ref, cls) in enumerate(dataset.samples):
        class_to_indices.setdefault(cls, []).append(i)

    # Single class mode: generate visualizations for all images of the given class
    if args.class_name:
        if args.class_name not in class_to_indices:
            raise ValueError(f"Class '{args.class_name}' not found in dataset. "
                             f"Available classes: {sorted(class_to_indices.keys())[:10]}...")

        indices = class_to_indices[args.class_name]
        class_dir = os.path.join(args.output_path, args.class_name.replace(' ', '_'))
        os.makedirs(class_dir, exist_ok=True)

        print(f"Generating visualizations for all {len(indices)} images of class '{args.class_name}'...")
        for i, idx in enumerate(indices):
            img = dataset.load_image(dataset.samples[idx][0])
            fname = f"{args.class_name.replace(' ', '_')}_{i:04d}.pdf"
            interpret(model, tokenizer, preprocess, img, args.class_name, concept_names,
                      A_matrix, args.backbone, args.device, class_dir, fname)
        print(f"Done! Saved {len(indices)} visualizations to '{class_dir}'")
        return

    # Default mode: random sampling from seen/unseen splits
    # Load Seen/Unseen Splits
    split_path = os.path.join(args.dataset_root, args.dataset, "embeddings/splits/class_split.pt")
    class_split = torch.load(split_path, map_location='cpu', weights_only=True)
    seen_classnames = [classnames[i] for i in class_split['seen_classes']]
    unseen_classnames = [classnames[i] for i in class_split['unseen_classes']]

    # Process a subset of classes
    def process_subset(subset_classnames, suffix):
        sampled_classes = random.sample(subset_classnames, min(args.num_classes, len(subset_classnames)))
        for cls_name in sampled_classes:
            if cls_name not in class_to_indices or not class_to_indices[cls_name]:
                continue # Skip if no images available for this class

            # Pick one random image for the visualization
            idx = random.choice(class_to_indices[cls_name])
            img = dataset.load_image(dataset.samples[idx][0])
            fname = f"{cls_name.replace(' ', '_')}_{suffix}.pdf"

            interpret(model, tokenizer, preprocess, img, cls_name, concept_names, A_matrix, args.backbone, args.device, args.output_path, fname)

    print(f"Generating image-level visualizations in '{args.output_path}'...")
    process_subset(seen_classnames, "seen")
    process_subset(unseen_classnames, "unseen")


if __name__ == "__main__":
    set_seed(1234)

    parser = argparse.ArgumentParser(description="Image-Level Analysis")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                        help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained A matrix checkpoint")
    parser.add_argument("--output_path", type=str, default="./image_level_outputs", 
                        help="Directory to save the visualizations")
    parser.add_argument("--num_classes", type=int, default=30, 
                        help="Number of classes to visualize per split")
    parser.add_argument("--class_name", type=str, default=None, 
                        help="Generate visualizations for all images of this class (skips random sampling)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")
    
    args = parser.parse_args()

    with torch.no_grad():
        main(args)