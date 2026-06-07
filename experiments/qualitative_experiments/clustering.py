import os
import argparse
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from PIL import Image

from dataset import RawImageDataset
from utils import (
    DATASET_CHOICES,
    set_seed, 
    load_backbone, 
    load_dataset_config, 
    get_text_embs
)


# ======================================================
# Dataloading
# ======================================================
def get_dataloader(preprocess, dataset_name, dataset_root, batch_size):
    dataset_obj = RawImageDataset(dataset_name, dataset_root, preprocess)
    loader = DataLoader(dataset_obj, batch_size=batch_size, num_workers=4, shuffle=False)
    return loader, dataset_obj


# ======================================================
# Visualization
# ======================================================
def visualize_cluster(concept_idx, target_concept, concept_scores, dataset_obj, output_path, image_size=224):
    """
    Display top-9 activating images for a given concept.
    """
    img_scores = concept_scores[:, concept_idx]
    topk_indices = torch.topk(img_scores, 9).indices.cpu().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(5.2, 5.8))
    fig.suptitle(f"Concept: {target_concept}", fontsize=14, fontweight="bold", y=0.96)

    for i, ax in enumerate(axes.flat):
        if i >= len(topk_indices):
            ax.axis("off")
            continue
        img = dataset_obj.load_image(dataset_obj.samples[topk_indices[i]][0])
        img = img.resize((image_size, image_size), Image.LANCZOS)
        ax.imshow(img)
        ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.1)
    plt.tight_layout(pad=0.2, rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_path, exist_ok=True)
    save_file = os.path.join(output_path, f"{target_concept.replace(' ', '_')}.pdf")
    plt.savefig(save_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Visualization saved to {save_file}")


# ======================================================
# Concept Score Calculation
# ======================================================
def calculate_concept_scores(model, tokenizer, backbone, loader, A_matrix, classnames, device):
    # Pre-compute class name embeddings once (not per batch)
    all_class_embs = get_text_embs(model, classnames, backbone, tokenizer, device)
    # Build a lookup from class name to index
    classname_to_idx = {name: i for i, name in enumerate(classnames)}

    concept_scores = None

    for batch in tqdm.tqdm(loader, desc="Computing concept activations"):
        batch_imgs, batch_classes = batch

        # Look up pre-computed embeddings for this batch
        batch_indices = [classname_to_idx[c] for c in batch_classes]
        class_embs = all_class_embs[batch_indices]

        batch_imgs = batch_imgs.to(device)
        img_embs = F.normalize(model.encode_image(batch_imgs), dim=1).float()

        # Compute activation scores using the EZPC A_matrix bottleneck
        batch_scores = (img_embs @ A_matrix) * (class_embs @ A_matrix)

        if concept_scores is None:
            concept_scores = batch_scores
        else:
            concept_scores = torch.cat((concept_scores, batch_scores), dim=0)

    # Normalize the scores
    concept_scores = concept_scores / concept_scores.max(dim=1, keepdim=True).values
    return concept_scores

# ===============================================================
# Main
# ===============================================================
def main(args):
    # Load backbone dynamically via utils
    model, preprocess, tokenizer = load_backbone(args.backbone, args.device)
    
    # Load A matrix weights onto the correct device
    A_matrix = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)

    # Load concept names and classnames using the universal config
    classnames, concept_names = load_dataset_config(args.dataset_root, args.dataset)

    if args.target_concept not in concept_names:
        raise ValueError(f"Concept '{args.target_concept}' not found in {args.dataset} concepts.")

    # Load data
    loader, dataset_obj = get_dataloader(preprocess, args.dataset, args.dataset_root, args.batch_size)

    # Calculate scores
    concept_scores = calculate_concept_scores(model, tokenizer, args.backbone, loader, A_matrix, classnames, args.device)

    # Visualize the clustering results
    concept_idx = concept_names.index(args.target_concept)
    visualize_cluster(concept_idx, args.target_concept, concept_scores, dataset_obj, args.output_path)


if __name__ == "__main__":
    set_seed(1234)

    parser = argparse.ArgumentParser(description="Concept Clustering Visualization")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                        help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained A matrix checkpoint")
    parser.add_argument("--target_concept", type=str, required=True, 
                        help="The concept to visualize")
    parser.add_argument("--output_path", type=str, default="./clustering_outputs", 
                        help="Directory to save the visualizations")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Dataloader batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")
    
    args = parser.parse_args()

    # Run the main function
    with torch.no_grad():
        main(args)