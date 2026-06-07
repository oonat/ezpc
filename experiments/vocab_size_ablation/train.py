import os
import argparse
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import EZPC
from utils import (
    DATASET_CHOICES,
    set_seed, 
    load_backbone, 
    backbone_to_name, 
    load_dataset_config, 
    load_train_embeddings, 
    get_text_embs
)

def optimize_param(ezpc_model, dataloader, class_name_embs, num_epochs, lr, device):
    optimizer = Adam(ezpc_model.parameters(), lr=lr)

    total_loss_list = []
    matching_loss_list, reconstruction_loss_list = [], []

    best_A = None
    best_loss = {'matching_loss': None, 'reconstruction_loss': None, 'total_loss': None}
    best_epoch = 0

    for epoch in range(num_epochs):
        epoch_matching_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_total_loss = 0.0

        for emb_batch in dataloader:
            emb_batch = emb_batch.to(device, non_blocking=True)

            matching_loss, reconstruction_loss, total_loss = ezpc_model(emb_batch, class_name_embs)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # Normalize A columns
            ezpc_model.normalize_weights()

            epoch_matching_loss += matching_loss.item()
            epoch_reconstruction_loss += reconstruction_loss.item()
            epoch_total_loss += total_loss.item()

        n_batches = len(dataloader)
        epoch_matching_loss /= n_batches
        epoch_reconstruction_loss /= n_batches
        epoch_total_loss /= n_batches

        matching_loss_list.append(epoch_matching_loss)
        reconstruction_loss_list.append(epoch_reconstruction_loss)
        total_loss_list.append(epoch_total_loss)

        if best_loss['total_loss'] is None or epoch_total_loss < best_loss['total_loss']:
            best_A = ezpc_model.A.clone().detach()
            best_epoch = epoch + 1
            best_loss.update({
                'matching_loss': epoch_matching_loss,
                'reconstruction_loss': epoch_reconstruction_loss,
                'total_loss': epoch_total_loss
            })

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Matching: {epoch_matching_loss:.6f} | "
              f"Reconstruction: {epoch_reconstruction_loss:.6f} | "
              f"Total: {epoch_total_loss:.6f}", flush=True)

    print(f"Final Matching Loss: {matching_loss_list[-1]:.6f}")
    print(f"Final Reconstruction Loss: {reconstruction_loss_list[-1]:.6f}")
    print(f"Final Total Loss: {total_loss_list[-1]:.6f}")

    print(f"Best Matching Loss: {best_loss['matching_loss']:.6f}")
    print(f"Best Reconstruction Loss: {best_loss['reconstruction_loss']:.6f}")
    print(f"Best Total Loss: {best_loss['total_loss']:.6f}")
    print(f"Best Epoch: {best_epoch}")

    return best_A, matching_loss_list, reconstruction_loss_list, total_loss_list

def subset_concepts(concept_names, num_concepts, seed):
    """Deterministic subset selection for concept vocabulary size experiment."""
    if num_concepts is None or num_concepts < 0 or num_concepts >= len(concept_names):
        return concept_names

    rng = random.Random(seed)
    idxs = list(range(len(concept_names)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:num_concepts])
    return [concept_names[i] for i in idxs]

def main(args):
    backbone_name = backbone_to_name(args.backbone)
    m_tag = "all" if args.num_concepts < 0 else str(args.num_concepts)
    subset_tag = f"m{m_tag}_seed{args.subset_seed}"

    output_folder = (
        f"{args.output_path}/{args.dataset}_backbone_{backbone_name}_weight_{args.lambda_weight}_"
        f"epoch_{args.num_epochs}_lr_{args.lr}_bs_{args.batch_size}_{subset_tag}"
    )
    os.makedirs(output_folder, exist_ok=True)

    # Load backbone
    model, _, tokenizer = load_backbone(args.backbone, args.device)

    # Get dataset config
    classnames, concept_names = load_dataset_config(args.dataset_root, args.dataset)

    # Load training embeddings
    train_img_tensor = load_train_embeddings(args.dataset_root, args.dataset, args.backbone)
    dataloader = DataLoader(train_img_tensor, batch_size=args.batch_size, shuffle=True, pin_memory=False)

    # Generate text embeddings
    class_name_embs = get_text_embs(model, classnames, args.backbone, tokenizer, args.device)

    # Subsample concepts based on m
    concept_names = subset_concepts(concept_names, args.num_concepts, args.subset_seed)

    # Save the exact concept list used
    with open(os.path.join(output_folder, "concept_names_used.txt"), "w") as f:
        for c in concept_names:
            f.write(c + "\n")
    print(f"Using m={len(concept_names)} concepts (seed={args.subset_seed}).")

    concept_matrix = get_text_embs(model, concept_names, args.backbone, tokenizer, args.device)

    # Init EZPC
    ezpc_model = EZPC(concept_matrix, args.lambda_weight).to(args.device)

    # Optimize params
    best_A, matching_loss_list, reconstruction_loss_list, total_loss_list = optimize_param(
        ezpc_model, dataloader, class_name_embs, args.num_epochs, args.lr, args.device
    )

    # Save weights and plot
    torch.save(best_A, os.path.join(output_folder, "best_A.pth"))

if __name__ == "__main__":
    set_seed(1234)

    parser = argparse.ArgumentParser(description="Concept vocabulary size (m) experiment - Training")
    parser.add_argument("--num_epochs", type=int, default=10000, 
                        help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--lambda_weight", type=float, default=1.0,
                        help="Lambda weight (reconstruction loss coefficient)")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                        help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--batch_size", type=int, default=1000000)
    parser.add_argument("--num_concepts", type=int, default=-1, help="m (number of concepts). Use -1 for all.")
    parser.add_argument("--subset_seed", type=int, default=1234)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")

    args = parser.parse_args()
    main(args)