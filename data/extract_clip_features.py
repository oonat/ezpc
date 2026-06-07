import os
import argparse
import tqdm
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision.datasets import Places365

from PIL import Image
from datasets import load_dataset
from huggingface_hub import login

from utils import (
    DATASET_CHOICES, 
    load_backbone,
    backbone_to_name
)

class CUBDataset(Dataset):
    def __init__(self, root_dir, split='train', preprocess=None):
        self.root_dir = root_dir
        self.split = split
        self.preprocess = preprocess

        # train/test split
        split_file = os.path.join(root_dir, 'train_test_split.txt')
        with open(split_file) as f:
            lines = f.readlines()
        self.is_train = {}
        for line in lines:
            img_id, is_train = line.strip().split()
            self.is_train[int(img_id)] = int(is_train)

        # image paths
        images_file = os.path.join(root_dir, 'images.txt')
        self.imgid_to_path = {}
        with open(images_file) as f:
            for line in f:
                img_id, path = line.strip().split()
                self.imgid_to_path[int(img_id)] = path

        # labels
        labels_file = os.path.join(root_dir, 'image_class_labels.txt')
        self.imgid_to_label = {}
        with open(labels_file) as f:
            for line in f:
                img_id, label = line.strip().split()
                self.imgid_to_label[int(img_id)] = int(label) - 1  # labels are 1-based in CUB

        # final split selection
        self.samples = []
        for img_id, is_train in self.is_train.items():
            if (split == 'train' and is_train) or (split == 'test' and not is_train):
                self.samples.append(img_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id = self.samples[idx]
        img_path = os.path.join(self.root_dir, 'images', self.imgid_to_path[img_id])
        image = Image.open(img_path).convert("RGB")
        if self.preprocess:
            image = self.preprocess(image)
        label = self.imgid_to_label[img_id]
        return image, label

class ImageNet100Dataset(Dataset):
    def __init__(self, root, wnids, preprocess):
        self.samples = []
        self.preprocess = preprocess
        wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

        for wnid in wnids:
            class_dir = os.path.join(root, wnid)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, wnid_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.preprocess(Image.open(path).convert("RGB"))
        return image, label

def _embed_loader(model, loader, device):
    emb_list, id_list = [], []
    for batch_imgs, batch_ids in tqdm.tqdm(loader):
        emb_list.append(model.encode_image(batch_imgs.to(device)).cpu())
        id_list.append(batch_ids)
    embs = F.normalize(torch.vstack(emb_list), dim=1)
    ids  = torch.cat(id_list, dim=0)
    return embs, ids

def _save_split(emb_dir, model_name, split, embs, ids):
    os.makedirs(emb_dir, exist_ok=True)
    torch.save(embs, os.path.join(emb_dir, f"{model_name}_{split}_embeddings.pt"))
    torch.save(ids,  os.path.join(emb_dir, f"{split}_ids.pt"))

def extract_cifar100_features(model, preprocess, args, model_name, device):
    dataset_dir = os.path.join(args.dataset_root, "CIFAR-100")

    # Training set
    trainset = torchvision.datasets.CIFAR100(
        root=dataset_dir, train=True,
        download=True, transform=preprocess
    )

    # Test set
    testset = torchvision.datasets.CIFAR100(
        root=dataset_dir, train=False,
        download=True, transform=preprocess
    )

    # Training loader
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )

    # Test loader
    testloader = DataLoader(
        testset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # Generate embeddings
    train_embs, train_ids = _embed_loader(model, trainloader, device)
    test_embs, test_ids = _embed_loader(model, testloader, device)

    # Save the embeddings
    emb_dir = os.path.join(dataset_dir, "embeddings")
    _save_split(emb_dir, model_name, "train", train_embs, train_ids)
    _save_split(emb_dir, model_name, "test", test_embs, test_ids)

def extract_cub_features(model, preprocess, args, model_name, device):
    dataset_dir = os.path.join(args.dataset_root, "CUB-200-2011")
    train_dataset = CUBDataset(dataset_dir, split='train', preprocess=preprocess)
    test_dataset = CUBDataset(dataset_dir, split='test', preprocess=preprocess)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Generate embeddings
    train_embs, train_ids = _embed_loader(model, trainloader, device)
    test_embs, test_ids = _embed_loader(model, testloader, device)

    # Save the embeddings
    emb_dir = os.path.join(dataset_dir, "embeddings")
    _save_split(emb_dir, model_name, "train", train_embs, train_ids)
    _save_split(emb_dir, model_name, "test", test_embs, test_ids)

def extract_imagenet_features(model, preprocess, args, model_name, device):
    dataset_dir = os.path.join(args.dataset_root, "ImageNet")
    train_ds = load_dataset("ILSVRC/imagenet-1k", split="train")
    val_ds = load_dataset("ILSVRC/imagenet-1k", split="validation")

    def transform_example(examples):
        examples['image'] = [preprocess(img) for img in examples['image']]
        return examples

    train_ds = train_ds.with_transform(transform_example)
    val_ds = val_ds.with_transform(transform_example)

    def collate_fn(batch):
        images = torch.stack([x["image"] for x in batch])
        labels_tensor = torch.tensor([x["label"] for x in batch])
        return images, labels_tensor

    trainloader = DataLoader(train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    testloader = DataLoader(val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # Generate embeddings
    train_embs, train_ids = _embed_loader(model, trainloader, device)
    test_embs, test_ids = _embed_loader(model, testloader, device)

    # Save the embeddings
    emb_dir = os.path.join(dataset_dir, "embeddings")
    _save_split(emb_dir, model_name, "train", train_embs, train_ids)
    _save_split(emb_dir, model_name, "test", test_embs, test_ids)

def extract_places365_features(model, preprocess, args, model_name, device):
    dataset_dir = os.path.join(args.dataset_root, "Places365")

    # Datasets
    trainset = Places365(
        root=dataset_dir,
        split="train-standard",
        small=True,
        download=False,
        transform=preprocess
    )
    valset = Places365(
        root=dataset_dir,
        split="val",
        small=True,
        download=False,
        transform=preprocess
    )

    # Loaders
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    testloader = DataLoader(
        valset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # Generate embeddings
    train_embs, train_ids = _embed_loader(model, trainloader, device)
    test_embs, test_ids = _embed_loader(model, testloader, device)

    # Save the embeddings
    emb_dir = os.path.join(dataset_dir, "embeddings")
    _save_split(emb_dir, model_name, "train", train_embs, train_ids)
    _save_split(emb_dir, model_name, "test", test_embs, test_ids)

def extract_imagenet100_features(model, preprocess, args, model_name, device):
    dataset_dir = os.path.join(args.dataset_root, "ImageNet-100")

    # Load label mapping
    with open(os.path.join(dataset_dir, "config", "Labels.json"), "r") as f:
        label_map = json.load(f)

    wnids = list(label_map.keys())

    # Training set
    trainset = ImageNet100Dataset(os.path.join(dataset_dir, "train"), wnids, preprocess)

    # Test set
    testset = ImageNet100Dataset(os.path.join(dataset_dir, "val"), wnids, preprocess)

    # Training loader
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )

    # Test loader
    testloader = DataLoader(
        testset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # Generate embeddings
    train_embs, train_ids = _embed_loader(model, trainloader, device)
    test_embs, test_ids = _embed_loader(model, testloader, device)

    # Save the embeddings
    emb_dir = os.path.join(dataset_dir, "embeddings")
    _save_split(emb_dir, model_name, "train", train_embs, train_ids)
    _save_split(emb_dir, model_name, "test", test_embs, test_ids)

def extract_features(args):
    # Model initialization
    print(f"Loading model: {args.backbone}")
    model_name = backbone_to_name(args.backbone)
    model, preprocess, _ = load_backbone(args.backbone, args.device)

    # Call the helper functions
    if args.dataset == "CIFAR-100":
        extract_cifar100_features(model, preprocess, args, model_name, args.device)
    elif args.dataset == "CUB-200-2011":
        extract_cub_features(model, preprocess, args, model_name, args.device)
    elif args.dataset == "ImageNet":
        login(token=os.environ.get("HF_TOKEN", ""))
        extract_imagenet_features(model, preprocess, args, model_name, args.device)
    elif args.dataset == "Places365":
        extract_places365_features(model, preprocess, args, model_name, args.device)
    elif args.dataset == "ImageNet-100":
        extract_imagenet100_features(model, preprocess, args, model_name, args.device)
    else:
        raise ValueError("Invalid dataset name!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for datasets")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                        help="Dataset to process")
    parser.add_argument("--dataset_root", type=str, default="./", 
                        help="Root directory containing the datasets")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader worker processes")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")

    args = parser.parse_args()

    with torch.no_grad():
        extract_features(args)
