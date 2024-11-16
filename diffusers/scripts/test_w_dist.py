import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
import ot
import argparse
from PIL import Image
from sklearn.covariance import EmpiricalCovariance

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and processor
model_id = "models/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def load_images(folder_path, batch_size=32):
    def safe_load_image(path):
        try:
            with Image.open(path) as img:
                return transforms.Resize((224, 224))(img.convert('RGB'))
        except:
            return None

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = ImageFolder(folder_path, transform=transform, loader=safe_load_image)
    dataset.samples = [(path, label) for path, label in dataset.samples if safe_load_image(path) is not None]
    dataset.imgs = dataset.samples
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

def get_embeddings(dataloader, model, processor):
    embeddings = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Processing images"):
            pil_images = [transforms.ToPILImage()(img) for img in images]
            inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
            outputs = model.get_image_features(**inputs)
            image_embeddings = outputs.cpu().numpy()
            embeddings.append(image_embeddings)
    
    return np.vstack(embeddings)

def compute_frechet_distance(embeddings1, embeddings2):
    """
    Calculate Fréchet distance between two sets of embeddings.
    This metric considers both mean and covariance of the distributions.
    """
    # Calculate mean and covariance for both distributions
    mu1, sigma1 = embeddings1.mean(axis=0), np.cov(embeddings1, rowvar=False)
    mu2, sigma2 = embeddings2.mean(axis=0), np.cov(embeddings2, rowvar=False)
    
    # Calculate squared difference between means
    diff = mu1 - mu2
    
    # Calculate matrix sqrt of the product of covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check if result is complex
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate Fréchet distance
    fd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    
    return fd

def compute_normalized_wasserstein(embeddings1, embeddings2):
    """
    Calculate Wasserstein distance normalized by the internal variance of each distribution.
    """
    # Compute covariance matrices
    cov1 = EmpiricalCovariance().fit(embeddings1)
    cov2 = EmpiricalCovariance().fit(embeddings2)
    
    # Compute whitening transformations
    whitening1 = np.linalg.inv(sqrtm(cov1.covariance_))
    whitening2 = np.linalg.inv(sqrtm(cov2.covariance_))
    
    # Whiten the embeddings
    white_embeddings1 = embeddings1.dot(whitening1)
    white_embeddings2 = embeddings2.dot(whitening2)
    
    # Calculate weights (uniform distribution)
    n1, n2 = len(embeddings1), len(embeddings2)
    weights1 = np.ones(n1) / n1
    weights2 = np.ones(n2) / n2
    
    # Compute cost matrix on whitened embeddings
    M = ot.dist(white_embeddings1, white_embeddings2)
    
    # Calculate normalized Wasserstein distance
    distance = ot.emd2(weights1, weights2, M)
    
    return distance

def compute_mahalanobis_distance(embeddings1, embeddings2):
    """
    Calculate average Mahalanobis distance between distributions.
    This metric considers the covariance structure of the reference distribution.
    """
    # Fit covariance of the first distribution (reference)
    cov = EmpiricalCovariance().fit(embeddings1)
    
    # Calculate means
    mean1 = np.mean(embeddings1, axis=0)
    mean2 = np.mean(embeddings2, axis=0)
    
    # Calculate Mahalanobis distance between means
    mahal_dist = np.sqrt(np.dot(np.dot((mean1 - mean2).T, 
                                      np.linalg.inv(cov.covariance_)), 
                                (mean1 - mean2)))
    
    return mahal_dist

def compute_distribution_metrics(embeddings1, embeddings2):
    """
    Compute multiple distribution shift metrics between two sets of embeddings.
    """
    # Normalize embeddings
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    metrics = {
        'frechet_distance': compute_frechet_distance(embeddings1, embeddings2),
        'normalized_wasserstein': compute_normalized_wasserstein(embeddings1, embeddings2),
        'mahalanobis_distance': compute_mahalanobis_distance(embeddings1, embeddings2)
    }
    
    return metrics

def main(folder_pairs, input_path):
    for folder1, folder2 in folder_pairs:
        print(f"\nComputing distribution shift metrics between {folder1} and {folder2}")
        
        # Load and process images from both folders
        dataloader1 = load_images(input_path + folder1, batch_size=64)
        dataloader2 = load_images(input_path + folder2, batch_size=64)
        
        # Get embeddings
        embeddings1 = get_embeddings(dataloader1, model, processor)
        embeddings2 = get_embeddings(dataloader2, model, processor)
        
        # Compute metrics
        metrics = compute_distribution_metrics(embeddings1, embeddings2)
        
        # Print results
        print(f"\nDistribution shift metrics for {folder1} vs {folder2}:")
        print(f"Fréchet Distance: {metrics['frechet_distance']:.4f}")
        print(f"Normalized Wasserstein Distance: {metrics['normalized_wasserstein']:.4f}")
        print(f"Mahalanobis Distance: {metrics['mahalanobis_distance']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate distribution shift metrics between CLIP embeddings")
    parser.add_argument("folder_pairs", nargs='+', help="Paths to the folder pairs, e.g., folder1A folder1B folder2A folder2B ...")
    parser.add_argument("--input", type=str, default="datasets/", 
                        help="Base input path containing the folders")
    args = parser.parse_args()

    # Group the folder paths into pairs
    folder_pairs = list(zip(args.folder_pairs[::2], args.folder_pairs[1::2]))
    
    main(folder_pairs, args.input)