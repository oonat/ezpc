import os
import argparse
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
    get_text_embs, 
    plot_loss
)

def optimize_param(ezpc_model, dataloader, class_name_embs, num_epochs, lr, device):
    optimizer = Adam(ezpc_model.parameters(), lr=lr)

    total_loss_list = []
    matching_loss_list, reconstruction_loss_list = [], []

    best_A = None
    best_loss = {'matching_loss': None, 'reconstruction_loss': None, 'total_loss': None}

    for epoch in range(num_epochs):
        epoch_matching_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_total_loss = 0.0

        for emb_batch in dataloader:
            emb_batch = emb_batch.to(device)

            matching_loss, reconstruction_loss, total_loss = ezpc_model(emb_batch, class_name_embs)

            optimizer.zero_grad()
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
            best_loss.update({
                'matching_loss': epoch_matching_loss,
                'reconstruction_loss': epoch_reconstruction_loss,
                'total_loss': epoch_total_loss
            })

        if epoch % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                f"Matching: {epoch_matching_loss:.6f} | "
                f"Reconstruction: {epoch_reconstruction_loss:.6f} | "
                f"Total: {epoch_total_loss:.6f}", flush=True)

    print(f"Final Matching Loss: {matching_loss_list[-1]}")
    print(f"Final Reconstruction Loss: {reconstruction_loss_list[-1]}")
    print(f"Final Total Loss: {total_loss_list[-1]}")

    print(f"Best Matching Loss: {best_loss['matching_loss']}")
    print(f"Best Reconstruction Loss: {best_loss['reconstruction_loss']}")
    print(f"Best Total Loss: {best_loss['total_loss']}")

    return best_A, matching_loss_list, reconstruction_loss_list, total_loss_list

def main(args):
    # Create a dir to save the checkpoint
    backbone_name = backbone_to_name(args.backbone)
    output_folder = (
        f"{args.output_path}/{args.dataset}_backbone_{backbone_name}"
        f"_weight_{args.lambda_weight}_epoch_{args.num_epochs}"
        f"_lr_{args.lr}_bs_{args.batch_size}"
    )
    os.makedirs(output_folder, exist_ok=True)

    # Load the CLIP/SigLIP backbone
    model, _, tokenizer = load_backbone(args.backbone, args.device)

    # Get the classnames and concepts
    classnames, concept_names = load_dataset_config(args.dataset_root, args.dataset)

    # Load the training img embeddings
    train_img_tensor = load_train_embeddings(args.dataset_root, args.dataset, args.backbone)
    dataloader = DataLoader(train_img_tensor, batch_size=args.batch_size, shuffle=True, pin_memory=False)

    # Generate classname and concept embeddings
    class_name_embs = get_text_embs(model, classnames, args.backbone, tokenizer, device=args.device)
    concept_matrix = get_text_embs(model, concept_names, args.backbone, tokenizer, device=args.device)

    # Init EZPC
    ezpc_model = EZPC(concept_matrix, args.lambda_weight).to(args.device)

    # Optimize params
    best_A, matching_loss_list, reconstruction_loss_list, total_loss_list = \
        optimize_param(ezpc_model, dataloader, class_name_embs, args.num_epochs, args.lr, args.device)

    # Save the weights
    torch.save(best_A, f"{output_folder}/best_A.pth")

    # Save the loss plots
    last_n_epochs = min(8000, args.num_epochs)
    start_epoch = args.num_epochs - last_n_epochs
    plot_loss(
        output_folder,
        matching_loss_list[start_epoch:],
        reconstruction_loss_list[start_epoch:],
        total_loss_list[start_epoch:],
        start_epoch
    )

if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(1234)

    parser = argparse.ArgumentParser(description="Training")
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
    parser.add_argument("--batch_size", type=int, default=1000000, 
                        help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")
    parser.add_argument("--output_path", type=str, default="./checkpoints", 
                        help="Directory for checkpoints")

    args = parser.parse_args()

    # Run the main function
    main(args)