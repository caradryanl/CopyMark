import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
from PIL import Image
import string



# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and processor from transformers
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
    
    # Filter out None values (failed loads) silently
    dataset.samples = [(path, label) for path, label in dataset.samples if safe_load_image(path) is not None]
    dataset.imgs = dataset.samples
    
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


def visualize_embeddings_3d(p, embeddings, labels, class_names, folder_names, folder_labels, ax):
    tsne = TSNE(n_components=3, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Find optimal hyperplane
    svm = SVC(kernel='linear')
    svm.fit(reduced_embeddings, folder_labels)

    # Create a mesh grid
    x_min, x_max = reduced_embeddings[:, 0].min() - 1, reduced_embeddings[:, 0].max() + 1
    y_min, y_max = reduced_embeddings[:, 1].min() - 1, reduced_embeddings[:, 1].max() + 1
    z_min, z_max = reduced_embeddings[:, 2].min() - 1, reduced_embeddings[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 20),
                         np.arange(y_min, y_max, (y_max - y_min) / 20))

    # Calculate corresponding z values for the hyperplane
    z_plane = (-svm.intercept_[0] - svm.coef_[0][0] * xx - svm.coef_[0][1] * yy) / svm.coef_[0][2]

    # Plot the hyperplane
    # ax.plot_surface(xx, yy, z_plane, alpha=0.2, color='gray')

    colors = ['#B4F8C8', '#F1D1D0']
    markers = ['o', 'o']  # Circle for points above, square for points below
    for i, folder in enumerate(folder_names):
        mask = folder_labels == i
        points = reduced_embeddings[mask]
        
        # Determine which points are above and below the plane
        z_points = points[:, 2]
        z_plane_at_points = (-svm.intercept_[0] - svm.coef_[0][0] * points[:, 0] - svm.coef_[0][1] * points[:, 1]) / svm.coef_[0][2]
        above_mask = z_points > z_plane_at_points
        below_mask = ~above_mask
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                c=colors[i], marker=markers[0], label=f"{folder[:-5]}", alpha=1, s=20)

        # Plot points above the plane
        # scatter_above = ax.scatter(points[above_mask, 0], points[above_mask, 1], points[above_mask, 2],
        #            c=colors[i], marker=markers[0], label=f"{folder[:-5]}", alpha=1, s=20)
        
        # # Plot points below the plane
        # scatter_below = ax.scatter(points[below_mask, 0], points[below_mask, 1], points[below_mask, 2],
        #            c=colors[i], marker=markers[1], label="", alpha=1, s=20)

    # Calculate metrics
    predictions = svm.predict(reduced_embeddings)
    cm = confusion_matrix(folder_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    # Print metrics
    print(f"\nMetrics for {folder_names[0]} vs {folder_names[1]}:")
    print(f"True Positive (TP): {tp}")
    print(f"False Positive (FP): {fp}")
    print(f"True Negative (TN): {tn}")
    print(f"False Negative (FN): {fn}")

    ax.legend(fontsize=18, markerscale=5, loc='lower left')
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_locator(plt.MaxNLocator(4))
    
    ax.tick_params(axis='both', which='major', labelsize=16)

def create_multiple_visualizations(all_embeddings_list, all_labels_list, all_classes_list, all_folders_list, folder_labels_list, output_path, title):
    plt.rcParams.update({'font.size': 16})  # Slightly reduced font size
    
    fig = plt.figure(figsize=(25, 5.5))  # Increased height slightly to accommodate the title
    
    for i in range(5):
        ax = fig.add_subplot(1, 5, i+1, projection='3d')
        visualize_embeddings_3d(i, all_embeddings_list[i], all_labels_list[i], all_classes_list[i], all_folders_list[i], folder_labels_list[i], ax)

        subplot_label = string.ascii_lowercase[i]
        ax.text2D(0.5, -0.1, f'({subplot_label})', transform=ax.transAxes, fontsize=20, fontweight='bold')
    
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)  # Reduced padding
    plt.subplots_adjust(top=0.92, bottom=0.16)  # Adjusted top margin to make room for the title
    fig.suptitle(title, fontsize=24, y=0.97)  # Add the overall title
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Use tight bounding box
    plt.close(fig)
    print(f"Visualization saved to {output_path}")

def main(folder_pairs, input_path, output_path, title):
    all_embeddings_list = []
    all_labels_list = []
    all_classes_list = []
    all_folders_list = []
    folder_labels_list = []

    for folder1, folder2 in folder_pairs:
        folders = [folder1, folder2]
        pair_embeddings = []
        pair_labels = []
        pair_classes = []
        pair_folder_labels = []

        for i, folder in enumerate(folders):
            folder_path = input_path + folder
            dataloader, classes = load_images(folder_path, batch_size=64)
            embeddings, labels = get_embeddings(dataloader, model, processor)
            pair_embeddings.append(embeddings)
            pair_labels.append(labels)
            pair_classes.extend(classes)
            pair_folder_labels.extend([i] * len(labels))

        all_folders_list.append(folders)
        all_embeddings_list.append(np.vstack(pair_embeddings))
        all_labels_list.append(np.concatenate(pair_labels))
        all_classes_list.append(pair_classes)
        folder_labels_list.append(np.array(pair_folder_labels))

    create_multiple_visualizations(all_embeddings_list, all_labels_list, all_classes_list, all_folders_list, folder_labels_list, output_path, title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CLIP embeddings of images from multiple pairs of folders")
    parser.add_argument("folder_pairs", nargs='+', help="Paths to the folder pairs, e.g., folder1A folder1B folder2A folder2B ...")
    parser.add_argument("--input", type=str, default="datasets/", 
                        help="Path to save the output visualization (default: embeddings_visualization_multiple.png)")
    parser.add_argument("--output", type=str, default="clip_embeddings_eval.png", 
                        help="Path to save the output visualization (default: embeddings_visualization_multiple.png)")
    parser.add_argument("--title", type=str, nargs='+', default=["CLIP", "Embeddings", "Visualization"], 
                        help="Overall title for the visualization (default: CLIP Embeddings Visualization)")
    args = parser.parse_args()

    # Group the folder paths into pairs
    folder_pairs = list(zip(args.folder_pairs[::2], args.folder_pairs[1::2]))

    if len(folder_pairs) != 5:
        raise ValueError("Please provide exactly 5 pairs of folders")
    
    title = ' '.join(args.title)

    main(folder_pairs, args.input, args.output, title)