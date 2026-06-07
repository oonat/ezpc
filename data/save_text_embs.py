import argparse
from pathlib import Path

import torch

from utils import (
    DATASET_CHOICES,
    set_seed,
    load_backbone,
    backbone_to_name,
    load_dataset_config,
    get_text_embs,
)


def main(args):
    embedding_dir = Path(f"{args.dataset_root}/{args.dataset}/embeddings")
    if not embedding_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {embedding_dir}")
    
    model_name = backbone_to_name(args.backbone)

    classname_path = embedding_dir / f"{model_name}_classname_embs.pt"
    concept_path = embedding_dir / f"{model_name}_concept_matrix.pt"

    if (classname_path.exists() or concept_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Cached embeddings already exist in {embedding_dir}. "
            f"Pass --overwrite to regenerate."
        )

    model, _, tokenizer = load_backbone(args.backbone, args.device)
    classnames, concept_names = load_dataset_config(args.dataset_root, args.dataset)

    print(f"Encoding {len(classnames)} classnames...")
    class_name_embs = get_text_embs(model, classnames, args.backbone, tokenizer, args.device)

    print(f"Encoding {len(concept_names)} concepts...")
    concept_matrix = get_text_embs(model, concept_names, args.backbone, tokenizer, args.device)

    torch.save(class_name_embs.cpu(), classname_path)
    torch.save(concept_matrix.cpu(), concept_path)

    print(f"Saved classname_embs.pt: {tuple(class_name_embs.shape)}")
    print(f"Saved concept_matrix.pt: {tuple(concept_matrix.shape)}")
    print(f"Location: {embedding_dir}")


if __name__ == "__main__":
    set_seed(1234)

    parser = argparse.ArgumentParser(
        description="Generate text embeddings for classnames and concepts"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                        help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone used for the checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing cached embeddings if present")

    args = parser.parse_args()
    main(args)
