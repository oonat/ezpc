import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import clip
import open_clip
import math
import json
import random


# ============================================================
# Dataset Configurations
# ============================================================
DATASET_CHOICES = ["CIFAR-100", "ImageNet-100", "CUB-200-2011", "Places365", "ImageNet"]

DATASET_CONFIG = {
    "CIFAR-100": {
        "concept_file": "config/cifar100_filtered.txt",
        "classname_source": {"type": "txt", "path": "config/cifar100_classes.txt"},
        "append_imagenet_concepts": True,
    },
    "CUB-200-2011": {
        "concept_file": "config/cub_filtered.txt",
        "classname_source": {"type": "txt", "path": "config/cub_classes.txt"},
        "append_imagenet_concepts": True,
    },
    "ImageNet": {
        "concept_file": "config/imagenet_filtered.txt",
        "classname_source": {"type": "json_values", "path": "config/class_mapping.json"},
        "append_imagenet_concepts": False,
    },
    "Places365": {
        "concept_file": "config/places365_filtered.txt",
        "classname_source": {"type": "txt", "path": "config/categories_places365_clean.txt"},
        "append_imagenet_concepts": True,
    },
    "ImageNet-100": {
        "concept_file": "config/imagenet_filtered.txt",
        "classname_source": {"type": "json_values", "path": "config/Labels.json"},
        "append_imagenet_concepts": False,
    },
}


# ============================================================
# Dataset Loaders
# ============================================================
def _load_classnames(dataset_root, dataset_name):
    """Load class names for a dataset."""
    cfg = DATASET_CONFIG[dataset_name]
    src = cfg["classname_source"]
    path = f"{dataset_root}/{dataset_name}/{src['path']}"

    if src["type"] == "txt":
        with open(path) as f:
            return [line.rstrip('\n') for line in f]
    elif src["type"] == "json_values":
        with open(path) as f:
            labels = json.load(f)
        return [label.split(",")[0] for label in labels.values()]

def _load_concept_names(dataset_root, dataset_name):
    # Load concept names for the given dataset
    cfg = DATASET_CONFIG[dataset_name]
    concept_path = f"{dataset_root}/{dataset_name}/{cfg['concept_file']}"

    with open(concept_path) as f:
        concept_names = [line.rstrip('\n') for line in f]

    # If configured, append ImageNet concepts
    if cfg["append_imagenet_concepts"]:
        imagenet_concept_path = f"{dataset_root}/ImageNet/config/imagenet_filtered.txt"
        with open(imagenet_concept_path) as f:
            imagenet_concepts = [line.rstrip('\n') for line in f]
        concept_names += imagenet_concepts

    # Filter the duplicates
    concept_names = list(set(concept_names))

    # Sort the concepts for reproducibility
    concept_names = sorted(concept_names)
    
    return concept_names

def load_dataset_config(dataset_root, dataset_name):
    """Load classnames and concept names for a dataset.
    Returns (classnames, concept_names).
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIG.keys())}")
    classnames = _load_classnames(dataset_root, dataset_name)
    concept_names = _load_concept_names(dataset_root, dataset_name)
    return classnames, concept_names

def load_train_embeddings(dataset_root, dataset_name, backbone):
    """Load seen-split training embeddings for a dataset."""
    bname = backbone_to_name(backbone)
    path = f"{dataset_root}/{dataset_name}/embeddings/splits/{bname}_seen_train_embs.pt"

    train_embs = torch.load(path, weights_only=True).float()
    train_embs = F.normalize(train_embs, dim=1)
    return train_embs

def load_test_embeddings(dataset_root, dataset_name, backbone):
    """Load seen/unseen test embeddings and targets, plus class split info."""
    bname = backbone_to_name(backbone)
    base = f"{dataset_root}/{dataset_name}/embeddings/splits"

    class_split = torch.load(f"{base}/class_split.pt", weights_only=True)

    seen_embs = torch.load(f"{base}/{bname}_seen_test_embs.pt", weights_only=True)
    seen_embs = F.normalize(seen_embs, dim=1).float()
    seen_targets = torch.load(f"{base}/seen_test_ids.pt", weights_only=True)

    unseen_embs = torch.load(f"{base}/{bname}_unseen_test_embs.pt", weights_only=True)
    unseen_embs = F.normalize(unseen_embs, dim=1).float()
    unseen_targets = torch.load(f"{base}/unseen_test_ids.pt", weights_only=True)

    return class_split, seen_embs, seen_targets, unseen_embs, unseen_targets


# ============================================================
# Backbone Loader
# ============================================================
SIGLIP_MODEL_ID = "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
SIGLIP_MODEL_NAME = "siglip-so400m-patch14-384"

def load_backbone(backbone, device):
    """Load a CLIP or SigLIP model. Returns (model, preprocess, tokenizer).
    tokenizer is None for CLIP models (uses clip.tokenize instead).
    """
    if _is_siglip(backbone):
        model, _, preprocess = open_clip.create_model_and_transforms(SIGLIP_MODEL_ID)
        tokenizer = open_clip.get_tokenizer(SIGLIP_MODEL_ID)
        model = model.to(device)
        model.eval()
        return model, preprocess, tokenizer
    else:
        model, preprocess = clip.load(backbone, device=device)
        model.eval()
        return model, preprocess, None

def _is_siglip(backbone):
    return "siglip" in backbone.lower()
    
def backbone_to_name(backbone):
    if _is_siglip(backbone):
        return SIGLIP_MODEL_NAME
    return backbone.replace("/", "-")


# ============================================================
# Embedding Generation
# ============================================================
def _encode_text(model, texts, backbone, tokenizer, device, batch_size=512):
    # Helper function to generate text embeddings
    siglip_check = _is_siglip(backbone)
    embeddings = []
    with torch.no_grad():
        for i in range(math.ceil(len(texts) / batch_size)):
            batch = texts[i * batch_size : (i + 1) * batch_size]
            if siglip_check:
                batch_tokens = tokenizer(batch).to(device)
            else:
                batch_tokens = clip.tokenize(batch).to(device)
            embeddings.append(model.encode_text(batch_tokens))
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = F.normalize(embeddings, dim=1).float()
    return embeddings.detach()

def get_text_embs(model, str_list, backbone, tokenizer, device):
    # Encode string list into normalized CLIP/SigLIP text embeddings
    texts = [f"a photo of {c}" for c in str_list]
    return _encode_text(model, texts, backbone, tokenizer, device)


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Plotting Utils
# ============================================================
def plot_loss(img_out_folder, matching_loss_list, reconstruction_loss_list, total_loss_list, start_epoch):  
    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(start_epoch, start_epoch + len(total_loss_list))
    
    # Plot total loss
    plt.plot(epochs, total_loss_list, label='Total Loss')
    
    # Add in a title and axes labels
    plt.title('Total Loss Plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Save the plot
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{img_out_folder}/total_loss_plot.png")
    plt.close()

    # Plot matching loss
    plt.plot(epochs, matching_loss_list, label='Matching Loss')

    # Add in a title and axes labels
    plt.title('Matching Loss Plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Save the plot
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{img_out_folder}/matching_loss_plot.png")
    plt.close()

    # Plot reconstruction loss
    plt.plot(epochs, reconstruction_loss_list, label='Reconstruction Loss')

    # Add in a title and axes labels
    plt.title('Reconstruction Loss Plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Save the plot
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{img_out_folder}/reconstruction_loss_plot.png")
    plt.close()


# ============================================================
# Fidelity Metrics
# ============================================================
def rowwise_pearson_corr(x, y, eps=1e-8):
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (x * y).sum(dim=1)
    den = torch.sqrt((x * x).sum(dim=1) * (y * y).sum(dim=1)).clamp_min(eps)
    return num / den

def spearman_corr_from_logits(a, b):
    a_ranks = a.argsort(dim=1).argsort(dim=1).float()
    b_ranks = b.argsort(dim=1).argsort(dim=1).float()
    return rowwise_pearson_corr(a_ranks, b_ranks)

def kl_div_from_logits(p_logits, q_logits, eps=1e-12):
    p = F.softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    return (p * ((p.clamp_min(eps)).log() - (q.clamp_min(eps)).log())).sum(dim=1)

def kendall_tau_topk_from_logits(clip_logits, ezpc_logits, topk=50):
    B, K = clip_logits.shape
    k = min(topk, K)

    top_idx = torch.topk(clip_logits, k=k, dim=1).indices
    clip_sub = torch.gather(clip_logits, 1, top_idx)
    ezpc_sub = torch.gather(ezpc_logits, 1, top_idx)

    clip_rank = clip_sub.argsort(dim=1).argsort(dim=1)
    ezpc_rank = ezpc_sub.argsort(dim=1).argsort(dim=1)

    clip_rank = clip_rank.unsqueeze(2)
    ezpc_rank = ezpc_rank.unsqueeze(2)

    d_clip = clip_rank - clip_rank.transpose(1, 2)
    d_ezpc = ezpc_rank - ezpc_rank.transpose(1, 2)

    iu = torch.triu_indices(k, k, offset=1, device=clip_logits.device)
    dc = d_clip[:, iu[0], iu[1]]
    de = d_ezpc[:, iu[0], iu[1]]

    discordant = ((dc * de) < 0).float().sum(dim=1)
    total_pairs = k * (k - 1) / 2.0
    tau = 1.0 - 2.0 * discordant / total_pairs
    return tau

# ============================================================
# Concept-Region Alignment Utils 
# ============================================================
class Layer4Hook:
    def __init__(self):
        self.feat = None

    def __call__(self, module, inp, out):
        self.feat = out

@torch.no_grad()
def get_patch_embeddings_rn50(model, preprocess, img, device):
    hook = Layer4Hook()
    handle = model.visual.layer4.register_forward_hook(hook)
    x = preprocess(img).unsqueeze(0).to(device)
    _ = model.encode_image(x)
    handle.remove()

    feat = hook.feat
    _, _, H, W = feat.shape
    patches = feat.permute(0, 2, 3, 1).reshape(H * W, 2048)

    attnpool = model.visual.attnpool
    patches = attnpool.v_proj(patches)
    patches = attnpool.c_proj(patches)
    patches = F.normalize(patches, dim=1).float()
    return patches, H, W

@torch.no_grad()
def get_patch_concept_acts(patch_embs, A):
    z = patch_embs @ A

    # Per-patch scale normalization
    denom = z.abs().amax(dim=1, keepdim=True) + 1e-8
    z_norm = z / denom

    # Mean-center across concepts per patch
    z_mc = z_norm - z_norm.mean(dim=1, keepdim=True)

    return z_mc

def heat_for_display(heat_2d):
    h = torch.relu(heat_2d.detach().float()).cpu().numpy()
    return h.astype(np.float32)