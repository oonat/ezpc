import os
import json
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    DATASET_CHOICES,
    set_seed, 
    load_backbone, 
    backbone_to_name, 
    load_dataset_config,
    load_test_embeddings,
    get_text_embs
)

# ============================================================
# Utility Functions
# ============================================================
def ensure_d_m_shape(A, Phi):
    """
    Ensures A and Φ both have shape (d, m).
    Handles (m, d) inputs by transposing.
    """
    if A.shape == Phi.shape:
        return A, Phi

    if A.shape[0] == Phi.shape[1] and A.shape[1] == Phi.shape[0]:
        print("Transposing A to shape (d, m).")
        A = A.t()
        return A, Phi

    if Phi.shape[0] == A.shape[1] and Phi.shape[1] == A.shape[0]:
        print("Transposing Φ to shape (d, m).")
        Phi = Phi.t()
        return A, Phi

    raise ValueError(f"Cannot reconcile shapes A={A.shape} and Φ={Phi.shape}")

def assert_unit_columns(tensor, name, atol=1e-2):
    """
    Assert every column of tensor is unit-norm.
    """
    col_norms = tensor.norm(dim=0)
    if not torch.allclose(col_norms, torch.ones_like(col_norms), atol=atol):
        raise AssertionError(
            f"{name} columns are not unit-norm "
            f"(min={col_norms.min():.4f}, max={col_norms.max():.4f}, "
            f"mean={col_norms.mean():.4f})"
        )

# ============================================================
# Alignment (A vs Φ)
# ============================================================
def compute_alignment(A, Phi, save_path):
    alignment = (A * Phi).sum(dim=0).cpu().numpy()

    plt.figure()
    plt.hist(alignment, bins=40)
    plt.title(r"Alignment: $\cos(A_j, \Phi_j)$")
    plt.xlabel("cosine similarity")
    plt.ylabel("# concepts")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "alignment_histogram.pdf"))
    plt.close()

    return {
        "mean": float(alignment.mean()),
        "median": float(np.median(alignment)),
        "std": float(alignment.std()),
        "min": float(alignment.min()),
        "max": float(alignment.max()),
    }

# ============================================================
# PCA Calculation
# ============================================================
def variance_stats(X, q=50, k=10):
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(Xc, q=q, center=False)
    var_per_pc = (S ** 2) / (X.shape[0] - 1)
    total_var = float(var_per_pc.sum())
    frac_topk = float(var_per_pc[:k].sum() / total_var)
    return total_var, frac_topk

def compute_pca(A, Phi, save_path):
    XA = A.t()  # (m, d)
    XPhi = Phi.t()

    XA_center = XA - XA.mean(dim=0, keepdim=True)
    XPhi_center = XPhi - XPhi.mean(dim=0, keepdim=True)

    UA, SA, VA = torch.pca_lowrank(XA_center, q=2)
    UPhi, SPhi, VPhi = torch.pca_lowrank(XPhi_center, q=2)

    plt.figure()
    plt.scatter(UPhi[:, 0].cpu(), UPhi[:, 1].cpu(), s=8, alpha=0.5, label=r"$\Phi$ (text)")
    plt.scatter(UA[:, 0].cpu(), UA[:, 1].cpu(), s=8, alpha=0.5, label="A (learned)")
    plt.title(r"PCA of Concept Space ($A$ vs $\Phi$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "pca_A_vs_Phi.pdf"))
    plt.close()

    total_var_A, frac_top10_A = variance_stats(XA)
    total_var_Phi, frac_top10_Phi = variance_stats(XPhi)

    return {
        "total_var_A": total_var_A,
        "total_var_Phi": total_var_Phi,
        "frac_top10_A": frac_top10_A,
        "frac_top10_Phi": frac_top10_Phi,
    }

# ============================================================
# Similarity Matrices
# ============================================================
def offdiag_stats(S):
    m = S.shape[0]
    mask = ~torch.eye(m, dtype=torch.bool)
    vals = S[mask]
    return float(vals.abs().mean()), float(vals.abs().std())

def compute_similarity_matrices(A, Phi, save_path):
    S_A = (A.t() @ A).cpu()
    S_Phi = (Phi.t() @ Phi).cpu()

    plt.figure()
    plt.imshow(S_Phi, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(r"Concept Similarity $\Phi^\top \Phi$")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sim_Phi.pdf"))
    plt.close()

    plt.figure()
    plt.imshow(S_A, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(r"Learned Similarity $A^\top A$")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sim_A.pdf"))
    plt.close()

    mu_off_Phi, std_off_Phi = offdiag_stats(S_Phi)
    mu_off_A, std_off_A = offdiag_stats(S_A)

    return {
        "mu_off_Phi": mu_off_Phi,
        "mu_off_A": mu_off_A,
        "std_off_Phi": std_off_Phi,
        "std_off_A": std_off_A,
    }

# ============================================================
# Activation Sparsity
# ============================================================
def compute_activation_sparsity(A, vx, ty, save_path, dataset_name, tau=0.01):
    """
    Computes concept sparsity over the provided image and text embeddings.
    """
    img_scores = vx @ A
    text_scores = ty @ A
    concept_scores = img_scores * text_scores

    mask = (concept_scores > tau)

    per_image = mask.sum(dim=1).cpu().numpy()
    per_concept = mask.sum(dim=0).cpu().numpy()

    plt.figure()
    plt.hist(per_image, bins=40)
    plt.title(f"Active Concepts per Image ({dataset_name})")
    plt.xlabel("# active concepts")
    plt.ylabel("# images")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "activation_sparsity_image.pdf"))
    plt.close()

    plt.figure()
    plt.hist(per_concept, bins=40)
    plt.title(f"Concept Usage Frequency ({dataset_name})")
    plt.xlabel("# images activated")
    plt.ylabel("# concepts")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "activation_sparsity_concept.pdf"))
    plt.close()

    return {
        "mean_active_per_image": float(per_image.mean()),
        "median_active_per_image": float(np.median(per_image)),
        "mean_images_per_concept": float(per_concept.mean()),
        "median_images_per_concept": float(np.median(per_concept)),
    }


# ===============================================================
# Main
# ===============================================================
def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    backbone_name = backbone_to_name(args.backbone)

    # Load backbone
    model, _, tokenizer = load_backbone(args.backbone, args.device)

    # Load A matrix
    print(f"Loading A matrix from {args.checkpoint_path}...")
    A = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True).float()

    # Get classnames and concepts
    classnames, concept_names = load_dataset_config(args.dataset_root, args.dataset)
    
    # Generate concept embeddings (Matrix Φ)
    print("Computing concept embeddings (Φ)...")
    Phi = get_text_embs(model, concept_names, args.backbone, tokenizer, args.device)

    # Ensure shapes align (Φ is (m, d) from get_text_embs, we need it as (d, m))
    A, Phi = ensure_d_m_shape(A, Phi.t())

    # Generate classname embeddings
    class_name_embs = get_text_embs(model, classnames, args.backbone, tokenizer, args.device)

    # Load precomputed image embeddings
    print(f"Loading precomputed test embeddings for {args.dataset}...")
    _, seen_embs, seen_targets, unseen_embs, unseen_targets = load_test_embeddings(
        args.dataset_root, args.dataset, args.backbone
    )
    
    # Combine all test data
    vx = torch.cat([seen_embs, unseen_embs], dim=0).to(args.device)
    targets = torch.cat([seen_targets, unseen_targets], dim=0).to(args.device)
    
    # Map target IDs to their respective text embeddings to get ty
    ty = class_name_embs[targets]

    # A and Φ need to have unit-norm columns
    assert_unit_columns(A, "A")
    assert_unit_columns(Phi, "Φ")

    # Run Structural Analyses
    print("Running alignment, PCA, and similarity analyses")
    alignment_stats = compute_alignment(A, Phi, args.output_path)
    pca_stats = compute_pca(A, Phi, args.output_path)
    sim_stats = compute_similarity_matrices(A, Phi, args.output_path)

    print("Running activation sparsity analysis")
    act_stats = compute_activation_sparsity(A, vx, ty, args.output_path, args.dataset)

    results = {
        "alignment": alignment_stats,
        "pca": pca_stats,
        "similarity": sim_stats,
        "activation": act_stats,
    }

    results_file = os.path.join(args.output_path, f"structure_analysis_{args.dataset}_{backbone_name}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved all results to: {results_file}")


if __name__ == "__main__":
    set_seed(1234)

    parser = argparse.ArgumentParser(description="Concept Space Structural Analysis")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                        help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained A matrix checkpoint")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--output_path", type=str, default="./structure_analysis_output", 
                        help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")

    args = parser.parse_args()

    with torch.no_grad():
        main(args)