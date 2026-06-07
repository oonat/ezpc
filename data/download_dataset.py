import os
import glob
import shutil
import argparse
import urllib.request
import tarfile
import zipfile
from tqdm import tqdm
from torchvision.datasets import CIFAR100, Places365

from utils import DATASET_CHOICES

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_archive(file_path, extract_path, cleanup=True):
    print(f"Extracting {file_path}...")
    if file_path.endswith("tar.gz") or file_path.endswith(".tgz") or file_path.endswith(".tar"):
        with tarfile.open(file_path, 'r:*') as tar:
            tar.extractall(path=extract_path)
    elif file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    print("Extraction complete!")

    if cleanup:
        remove_archives(file_path)

def remove_archives(*paths):
    """Delete downloaded archive files once they have been extracted."""
    for path in paths:
        if os.path.isfile(path):
            print(f"Removing archive {path}")
            os.remove(path)

def setup_cifar100(dataset_root):
    print("\n--- Downloading CIFAR-100 ---")
    cifar_dir = os.path.join(dataset_root, "CIFAR-100")
    os.makedirs(cifar_dir, exist_ok=True)
    CIFAR100(root=cifar_dir, train=True, download=True)
    CIFAR100(root=cifar_dir, train=False, download=True)
    # torchvision leaves the downloaded tarball next to the extracted folder
    remove_archives(os.path.join(cifar_dir, "cifar-100-python.tar.gz"))
    print("CIFAR-100 ready.")

def setup_cub200(dataset_root):
    print("\n--- Downloading CUB-200-2011 ---")
    cub_dir = os.path.join(dataset_root, "CUB-200-2011")
    os.makedirs(cub_dir, exist_ok=True)
    
    # Images and Annotations
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    tgz_path = os.path.join(cub_dir, "CUB_200_2011.tgz")
    images_dir = os.path.join(cub_dir, "images")

    if not os.path.exists(images_dir):
        print("Fetching Main Dataset...")
        download_url(url, tgz_path)
        extract_archive(tgz_path, cub_dir)

        nested_dir = os.path.join(cub_dir, "CUB_200_2011")
        if os.path.exists(nested_dir):
            print("Flattening directory structure...")
            for item in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, item), os.path.join(cub_dir, item))
            os.rmdir(nested_dir) # Remove the empty nested folder
    else:
        print("CUB-200-2011 images already extracted. Skipping download.")

    # Segmentations
    seg_url = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz"
    seg_tgz_path = os.path.join(cub_dir, "segmentations.tgz")
    seg_dir = os.path.join(cub_dir, "segmentations")

    if not os.path.exists(seg_dir):
        print("Fetching Segmentations...")
        download_url(seg_url, seg_tgz_path)
        extract_archive(seg_tgz_path, cub_dir)
    else:
        print("CUB-200-2011 segmentations already extracted. Skipping download.")

    print("CUB-200-2011 ready!")

def setup_places365(dataset_root):
    print("\n--- Downloading Places365 ---")
    places_dir = os.path.join(dataset_root, "Places365")
    os.makedirs(places_dir, exist_ok=True)
    
    try:
        print("Fetching Validation Set...")
        Places365(root=places_dir, split='val', small=True, download=True)
        
        print("Places365 train set size is too large. To download it, uncomment the line below.")
        #Places365(root=places_dir, split='train-standard', small=True, download=True)

        # torchvision leaves the downloaded .tar archives after extraction
        remove_archives(*glob.glob(os.path.join(places_dir, "*.tar")))
        print("Places365 ready!")
    except RuntimeError as e:
        print(f"\nError downloading Places365: {e}")

def print_imagenet_instructions():
    print("\n" + "="*50)
    print("MANUAL ACTION REQUIRED: ImageNet-1k & ImageNet-100")
    print("="*50)
    print("ImageNet is a gated dataset and is not downloaded by this script.")
    print("\n--- ImageNet-1k ---")
    print("Loaded automatically from HuggingFace ('ILSVRC/imagenet-1k') during")
    print("feature extraction. Before running extract_clip_features.py:")
    print("  1. Create a HuggingFace account.")
    print("  2. Accept the dataset terms at:")
    print("     https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    print("  3. Authenticate locally: `huggingface-cli login`")
    print("No manual image-net.org download is required.")
    print("\n--- ImageNet-100 ---")
    print("A 100-class subset, loaded from local folders (not auto-downloaded).")
    print("We use the Kaggle ImageNet-100 release:")
    print("  https://www.kaggle.com/datasets/ambityga/imagenet100")
    print("After downloading, arrange it under '<dataset_root>/ImageNet-100/' as:")
    print("  ImageNet-100/")
    print("    config/Labels.json        # 100 wnid -> class-name mapping (provided)")
    print("    train/<wnid>/*.JPEG       # merge train.X1, train.X2, train.X3, train.X4")
    print("    val/<wnid>/*.JPEG         # rename val.X -> val")
    print("i.e. merge the four train.X* folders into a single 'train/', and rename")
    print("'val.X' to 'val/'. The 100 wnids are the keys of config/Labels.json.")
    print("="*50 + "\n")

def main(args):
    os.makedirs(args.dataset_root, exist_ok=True)
    
    if args.dataset == "CIFAR-100":
        setup_cifar100(args.dataset_root)
    elif args.dataset == "CUB-200-2011":
        setup_cub200(args.dataset_root)
    elif args.dataset == "Places365":
        setup_places365(args.dataset_root)
    elif args.dataset in ["ImageNet", "ImageNet-100"]:
        print_imagenet_instructions()
    else:
        print(f"Unknown dataset: {args.dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EZPC Datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES, 
                        help="The dataset to download or setup")
    parser.add_argument("--dataset_root", type=str, default="./", 
                        help="Root directory to save datasets")
    args = parser.parse_args()
    
    main(args)