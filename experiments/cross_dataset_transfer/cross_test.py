import os
import argparse
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    DATASET_CHOICES,
    set_seed, 
    load_backbone, 
    backbone_to_name, 
    load_dataset_config, 
    get_text_embs
)

def get_dataloader(seen_class_embs, seen_class_targets, unseen_class_embs, unseen_class_targets, batch_size=512):
    seen_dataset = TensorDataset(seen_class_embs, seen_class_targets)
    seen_loader = DataLoader(seen_dataset, batch_size=batch_size, num_workers=4)

    unseen_dataset = TensorDataset(unseen_class_embs, unseen_class_targets)
    unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, num_workers=4)

    return seen_loader, unseen_loader

def test(val_loader, class_name_embs, A_matrix, device, shift=None):
    clip_true_count = 0
    ezpc_true_count = 0
    total_count = 0

    for batch in tqdm.tqdm(val_loader, desc="Evaluating"):
        batch_embs, batch_ids = batch
        if shift is not None:
            batch_ids += shift
            
        batch_embs, batch_ids = batch_embs.to(device), batch_ids.to(device)
        batch_embs = F.normalize(batch_embs, dim=1).float()

        clip_sim_scores = batch_embs @ class_name_embs.T
        clip_preds = torch.argmax(clip_sim_scores, dim=1)

        ezpc_sim_scores = batch_embs @ A_matrix @ A_matrix.T @ class_name_embs.T
        ezpc_preds = torch.argmax(ezpc_sim_scores, dim=1)

        clip_true_count += torch.sum(clip_preds == batch_ids).item()
        ezpc_true_count += torch.sum(ezpc_preds == batch_ids).item()
        total_count += batch_embs.shape[0]

    clip_accuracy = clip_true_count / total_count
    ezpc_accuracy = ezpc_true_count / total_count

    return clip_accuracy, ezpc_accuracy

def load_test_data(dataset_root, dataset_name, backbone_name):
    emb_path = os.path.join(dataset_root, dataset_name, "embeddings", f"{backbone_name}_test_embeddings.pt")
    target_path = os.path.join(dataset_root, dataset_name, "embeddings", "test_ids.pt")
    
    test_img_tensor = torch.load(emb_path, map_location="cpu", weights_only=True).float()
    test_targets = torch.load(target_path, map_location="cpu", weights_only=True)
    
    classnames, _ = load_dataset_config(dataset_root, dataset_name)
    
    return test_img_tensor, test_targets, classnames

# ===============================================================
# Main
# ===============================================================
def main(args):
    # Folder path to load weights
    backbone_name = backbone_to_name(args.backbone)
    out_folder = (f"{args.output_path}/source-dataset_{args.source_dataset}_"
                  f"target-dataset_{args.target_dataset}_"
                  f"backbone_{backbone_name}_"
                  f"weight_{args.lambda_weight}_"
                  f"epoch_{args.num_epochs}_"
                  f"lr_{args.lr}_bs_{args.batch_size}")
    
    # Init the backbone 
    model, _, tokenizer = load_backbone(args.backbone, args.device)

    # Load the trained checkpoint
    A_matrix = torch.load(os.path.join(out_folder, "best_A.pth"), map_location=args.device, weights_only=True)

    # Load seen and unseen datasets using the universal config
    seen_test_img_tensor, seen_class_targets, seen_classnames = load_test_data(args.dataset_root, args.source_dataset, backbone_name)
    unseen_test_img_tensor, unseen_class_targets, unseen_classnames = load_test_data(args.dataset_root, args.target_dataset, backbone_name)

    # Get classname embeddings dynamically via get_text_embs
    all_classnames = seen_classnames + unseen_classnames
    all_classname_embs = get_text_embs(model, all_classnames, args.backbone, tokenizer, args.device)
    seen_classname_embs = get_text_embs(model, seen_classnames, args.backbone, tokenizer, args.device)
    unseen_classname_embs = get_text_embs(model, unseen_classnames, args.backbone, tokenizer, args.device)

    # Get dataloaders
    seen_loader, unseen_loader = get_dataloader(
        seen_test_img_tensor, seen_class_targets, 
        unseen_test_img_tensor, unseen_class_targets,
        batch_size=args.batch_size
    )

    with torch.no_grad():
        seen_clip_acc, seen_ezpc_acc = test(seen_loader, seen_classname_embs, A_matrix, args.device)
        unseen_clip_acc, unseen_ezpc_acc = test(unseen_loader, unseen_classname_embs, A_matrix, args.device)

        seen_g_clip_acc, seen_g_ezpc_acc = test(seen_loader, all_classname_embs, A_matrix, args.device)
        unseen_g_clip_acc, unseen_g_ezpc_acc = test(unseen_loader, all_classname_embs, A_matrix, args.device, shift=len(seen_classnames))

        clip_harmonic_mean = 2 * seen_g_clip_acc * unseen_g_clip_acc / max(seen_g_clip_acc + unseen_g_clip_acc, 1e-8)
        ezpc_harmonic_mean = 2 * seen_g_ezpc_acc * unseen_g_ezpc_acc / max(seen_g_ezpc_acc + unseen_g_ezpc_acc, 1e-8)

    results = {
        "CLIP SEEN Accuracy": seen_clip_acc,
        "CLIP UNSEEN Accuracy": unseen_clip_acc,
        "CLIP Generalized SEEN Accuracy": seen_g_clip_acc,
        "CLIP Generalized UNSEEN Accuracy": unseen_g_clip_acc,
        "CLIP Generalized Harmonic Mean": clip_harmonic_mean,
        "EZPC SEEN Accuracy": seen_ezpc_acc,
        "EZPC UNSEEN Accuracy": unseen_ezpc_acc,
        "EZPC Generalized SEEN Accuracy": seen_g_ezpc_acc,
        "EZPC Generalized UNSEEN Accuracy": unseen_g_ezpc_acc,
        "EZPC Generalized Harmonic Mean": ezpc_harmonic_mean
    }

    # Save to txt file
    result_file = os.path.join(out_folder, "results.txt")
    with open(result_file, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"\nResults saved to {result_file}")

if __name__ == "__main__":
    set_seed(1234)

    parser = argparse.ArgumentParser(description="Cross Dataset Testing")
    parser.add_argument("--num_epochs", type=int, default=10000, 
                        help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--lambda_weight", type=float, default=1.0,
                        help="Lambda weight (reconstruction loss coefficient)")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--source_dataset", type=str, required=True, choices=DATASET_CHOICES, 
                        help="Source dataset name")
    parser.add_argument("--target_dataset", type=str, required=True, choices=DATASET_CHOICES, 
                        help="Target dataset name")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--batch_size", type=int, default=512, 
                        help="Batch size for testing")
    parser.add_argument("--output_path", type=str, default="./outputs", 
                        help="Directory where models are saved")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")

    args = parser.parse_args()

    main(args)