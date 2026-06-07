import os
import glob
import json
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, Places365
from huggingface_hub import login

from datasets import load_dataset

class EvalDataset(Dataset):
    """Dataset for evaluation: maps embeddings to class name strings."""
    def __init__(self, samples, targets, classnames):
        self.samples = samples
        self.targets = targets
        self.classnames = classnames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]
        target_classname = self.classnames[target]
        return sample, target_classname

class RawImageDataset(Dataset):
    """
    Unified dataset interface for multiple benchmarks.
    Images are loaded lazily in __getitem__ to avoid OOM on large datasets.
    """
    def __init__(self, dataset_name, dataset_root, preprocess):
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.preprocess = preprocess
        # Store (path_or_ref, class_name) tuples instead of loaded images
        self.samples = []

        if dataset_name == "CIFAR-100":
            self._cifar_ds = CIFAR100(root=f"{dataset_root}/CIFAR-100", train=False, download=True)
            self.samples = [(i, self._cifar_ds.classes[label].replace("_", " ")) for i, (_, label) in enumerate(self._cifar_ds)]

        elif dataset_name == "CUB-200-2011":
            base = f"{dataset_root}/CUB-200-2011/images"
            paths = sorted(glob.glob(f"{base}/**/*.jpg", recursive=True))
            # Extracts the class name and strips any prepended numbers (e.g., "001.Black_footed_Albatross")
            self.samples = [(p, p.split("/")[-2].replace("_", " ").split(".")[-1]) for p in paths]

        elif dataset_name == "ImageNet-100":
            base = f"{dataset_root}/ImageNet-100/val"
            with open(f"{dataset_root}/ImageNet-100/config/Labels.json") as f:
                labels = json.load(f)
            paths = sorted(glob.glob(f"{base}/**/*.JPEG", recursive=True))
            self.samples = [(p, labels[p.split("/")[-2]].split(",")[0]) for p in paths]

        elif dataset_name == "ImageNet":
            login(token=os.environ.get("HF_TOKEN", ""))
            self._hf_ds = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=f"{dataset_root}/ImageNet")
            self._label_map = self._hf_ds.features["label"].int2str
            self.samples = [(i, self._label_map(s["label"]).split(",")[0].strip()) for i, s in enumerate(self._hf_ds)]

        elif dataset_name == "Places365":
            self._places_ds = Places365(root=f"{dataset_root}/Places365", split="val",
                               small=True, download=False, transform=None)
            with open(f"{dataset_root}/Places365/config/categories_places365_clean.txt") as f:
                classes = [line.rstrip() for line in f]
            self.samples = [(i, classes[label]) for i, (_, label) in enumerate(self._places_ds)]

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def load_image(self, ref):
        """Load a PIL image from a reference (file path or dataset index)."""
        if self.dataset_name == "CIFAR-100":
            img, _ = self._cifar_ds[ref]
            return img.convert("RGB")
        elif self.dataset_name in ("CUB-200-2011", "ImageNet-100"):
            return Image.open(ref).convert("RGB")
        elif self.dataset_name == "ImageNet":
            return self._hf_ds[ref]["image"].convert("RGB")
        elif self.dataset_name == "Places365":
            img, _ = self._places_ds[ref]
            return img.convert("RGB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ref, class_name = self.samples[idx]
        img = self.load_image(ref)
        return self.preprocess(img), class_name
