import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and processor from transformers
model_id = "models/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def load_images(folder_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageFolder(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader, dataset.classes

def get_embeddings(dataloader, model, processor):
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels in tqdm(dataloader, desc="Processing images"):
            pil_images = [transforms.ToPILImage()(img) for img in images]
            inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
            outputs = model.get_image_features(**inputs)
            image_embeddings = outputs.cpu().numpy()
            embeddings.append(image_embeddings)
            labels.extend(batch_labels.tolist())
    
    return np.vstack(embeddings), np.array(labels)

def visualize_embeddings_3d(embeddings, labels, class_names, folder_labels, title, output_path):
    tsne = TSNE(n_components=3, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'blue']
    for i, folder in enumerate(['Folder 1', 'Folder 2']):
        mask = folder_labels == i
        scatter = ax.scatter(reduced_embeddings[mask, 0], 
                             reduced_embeddings[mask, 1], 
                             reduced_embeddings[mask, 2],
                             c=colors[i], label=folder, alpha=0.6)
    
    ax.set_title(title)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_zlabel("t-SNE dimension 3")
    
    ax.legend()
    
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('viridis', len(unique_labels))
    norm = plt.Normalize(unique_labels.min(), unique_labels.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Classes', ticks=unique_labels)
    cbar.set_ticklabels([class_names[l] for l in unique_labels])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Visualization saved to {output_path}")

def main(folder1, folder2, output_path):
    folders = [folder1, folder2]
    all_embeddings = []
    all_labels = []
    all_classes = []
    folder_labels = []

    for i, folder in enumerate(folders):
        dataloader, classes = load_images(folder, batch_size=64)
        embeddings, labels = get_embeddings(dataloader, model, processor)
        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_classes.extend(classes)
        folder_labels.extend([i] * len(labels))

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)
    folder_labels = np.array(folder_labels)

    visualize_embeddings_3d(all_embeddings, all_labels, all_classes, folder_labels, 
                            "Image Embeddings Visualization (3D)", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CLIP embeddings of images from two folders")
    parser.add_argument("folder1", type=str, help="Path to the first folder of images")
    parser.add_argument("folder2", type=str, help="Path to the second folder of images")
    parser.add_argument("--output", type=str, default="embeddings_visualization.png", 
                        help="Path to save the output visualization (default: embeddings_visualization.png)")
    args = parser.parse_args()

    main(args.folder1, args.folder2, args.output)