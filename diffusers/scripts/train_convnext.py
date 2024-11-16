import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from timm import create_model
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image
from tqdm import tqdm
import textwrap

plt.rcParams.update({'font.size': 14})  # Increase base font size


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.label = os.path.basename(root_dir)  # Use folder name as the label

        image_folder = os.path.join(root_dir, 'images')
        for img_name in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_name)
            if self.is_valid_image(img_path):
                self.images.append(img_path)

    def is_valid_image(self, img_path):
        try:
            with Image.open(img_path) as img:
                # Instead of img.verify(), we'll attempt to load the image data
                img.load()
            return True
        except:
            print(f"Skipping invalid image: {img_path}.")
            return False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.label

def safe_transform(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def load_data(folder1, folder2, batch_size=32):
    dataset1 = CustomImageDataset(folder1, transform=safe_transform)
    dataset2 = CustomImageDataset(folder2, transform=safe_transform)

    print(f"Dataset 1 label: {dataset1.label}")
    print(f"Dataset 2 label: {dataset2.label}")
    print(f"Dataset 1 size: {len(dataset1)}")
    print(f"Dataset 2 size: {len(dataset2)}")

    combined_dataset = ConcatDataset([dataset1, dataset2])
    
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader, dataset1.label, dataset2.label

def train_model(model, dataloader, criterion, optimizer, label1, label2, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Wrap dataloader with tqdm for progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = torch.tensor([0 if label == label1 else 1 for label in labels]).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar description with current loss
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

def evaluate_model(model, dataloader, label1, label2):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Handle both 1D and 2D outputs
            if outputs.dim() == 2 and outputs.shape[1] == 2:
                logits = outputs[:, 1]  # Take the positive class logit
            else:
                logits = outputs  # The output is already 1D
            
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend([0 if label == label1 else 1 for label in labels])
    
    return np.array(all_logits), np.array(all_labels)

def find_threshold(logits, labels, target_fpr, start=0, end=10.0, step=0.01):
    best_threshold = start
    best_fpr = 1.0
    best_tpr = 0.0
    end = logits.max()
    
    for threshold in np.arange(start, end, step):
        preds = (logits > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if fpr <= target_fpr and tpr > best_tpr:
            best_threshold = threshold
            best_fpr = fpr
            best_tpr = tpr
    
    return {
        'threshold': best_threshold,
        'fpr': best_fpr,
        'tpr': best_tpr
    }

def evaluate_threshold(labels, preds, threshold):
    # Convert predictions to binary based on the threshold
    binary_preds = (preds >= threshold).astype(int)
    
    # Ensure labels are also binary
    binary_labels = labels.astype(int)
    
    # Now compute the confusion matrix
    tn, fp, fn, tp = confusion_matrix(binary_labels, binary_preds).ravel()
    
    # Calculate TPR and FPR
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return tpr, fpr


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path):
    model = create_model('convnext_tiny', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model


def plot_roc_curve(train_fpr, train_tpr, train_roc_auc, eval_fpr, eval_tpr, eval_roc_auc, train_folder1, train_folder2, eval_folder1, eval_folder2, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(train_fpr, train_tpr, color='yellowgreen', lw=4, label=f'Val: AUC = {train_roc_auc:.4f}')
    plt.plot(eval_fpr, eval_tpr, color='salmon', lw=4, label=f'Test: AUC = {eval_roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=4, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=24)
    plt.ylabel('TPR', fontsize=24)
    plt.legend(loc="lower right", fontsize=26)

    if 'hakubooru' in train_folder1:
        title = "hakubooru: member vs non-member"
    elif 'laion-mi' in train_folder1:
        title = "laion-mi: member vs non-member"
    else:
        title = f"{os.path.basename(train_folder1)[:-5]} vs {os.path.basename(train_folder2)[:-5]}"
    wrapped_title = "\n".join(textwrap.wrap(title, width=60))
    plt.title(wrapped_title, fontsize=26, fontweight='bold')


    plt.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.grid(which='major', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)

    plt.subplots_adjust(top=0.90, bottom=0.16)  # Adjusted top margin to make room for the title
    plt.savefig(output_path, dpi=300)
    plt.close()


def main(input, train_folder1, train_folder2, eval_folders, num_epochs, batch_size, learning_rate, output_path, model_path, load_saved_model):
    # Load training data
    train_dataloader, label1, label2 = load_data(input + train_folder1, input + train_folder2, batch_size)
    
    if load_saved_model and os.path.exists(model_path):
        # Load the saved model
        model = load_model(model_path)
    else:
        # Create and train a new model
        model = create_model('convnext_tiny', pretrained=True, num_classes=2)
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        train_model(model, train_dataloader, criterion, optimizer, label1, label2, num_epochs)
        
        # Save the trained model
        save_model(model, model_path)
    
    logits, labels = evaluate_model(model, train_dataloader, label1, label2)
    
    # Print logits statistics
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{np.min(logits):.4f}, {np.max(logits):.4f}]")
    print(f"Logits mean: {np.mean(logits):.4f}")
    print(f"Logits std: {np.std(logits):.4f}")
    
    # Calculate AUC for training data
    train_fpr, train_tpr, _ = roc_curve(labels, logits)
    train_roc_auc = auc(train_fpr, train_tpr)
    print(f"Training AUC: {train_roc_auc:.4f}")

    # Find thresholds
    threshold_1percent = find_threshold(logits, labels, 0.01)
    threshold_01percent = find_threshold(logits, labels, 0.001)
    
    print(f"Threshold for 1% FPR: {threshold_1percent['threshold']:.4f}")
    print(f"TPR at 1% FPR: {threshold_1percent['tpr']:.4f}")
    print(f"Threshold for 0.1% FPR: {threshold_01percent['threshold']:.4f}")
    print(f"TPR at 0.1% FPR: {threshold_01percent['tpr']:.4f}")

    # Evaluate on the single pair of evaluation folders
    eval_folder1, eval_folder2 = eval_folders[0]
    eval_dataloader, eval_label1, eval_label2 = load_data(input + eval_folder1, input + eval_folder2, batch_size)

    print("\nEvaluating on evaluation folder pair:")
    eval_logits, eval_labels = evaluate_model(model, eval_dataloader, eval_label1, eval_label2)

    # Calculate AUC for evaluation data
    eval_fpr, eval_tpr, _ = roc_curve(eval_labels, eval_logits)
    eval_roc_auc = auc(eval_fpr, eval_tpr)
    print(f"Evaluation AUC: {eval_roc_auc:.4f}")

    tpr_1percent, fpr_1percent = evaluate_threshold(eval_labels, eval_logits, threshold_1percent['threshold'])
    print(f"1% threshold - TPR: {tpr_1percent:.4f}, FPR: {fpr_1percent:.4f}")
    tpr_01percent, fpr_01percent = evaluate_threshold(eval_labels, eval_logits, threshold_01percent['threshold'])
    print(f"0.1% threshold - TPR: {tpr_01percent:.4f}, FPR: {fpr_01percent:.4f}")

    # Plot ROC curves for both training and evaluation data
    plot_roc_curve(train_fpr, train_tpr, train_roc_auc, 
                   eval_fpr, eval_tpr, eval_roc_auc, 
                   train_folder1, train_folder2, 
                   eval_folder1, eval_folder2, 
                   output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvNeXt model and find optimal thresholds")
    parser.add_argument("train_folder1", type=str, help="Path to the first training folder")
    parser.add_argument("train_folder2", type=str, help="Path to the second training folder")
    parser.add_argument("eval_folders", nargs='+', help="Paths to additional folder pairs for evaluation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="roc_curve.png", help="Path to save the ROC curve")
    parser.add_argument("--model_path", type=str, default="convnext_model.pth", help="Path to save/load the model")
    parser.add_argument("--load_model", action="store_true", help="Load a saved model instead of training a new one")
    parser.add_argument("--input", type=str, default="datasets/", 
                        help="Path to save the output visualization (default: embeddings_visualization_multiple.png)")
    
    args = parser.parse_args()
    
    # Ensure even number of evaluation folders
    if len(args.eval_folders) != 2:
        raise ValueError("Exactly two evaluation folders must be provided")
    
    eval_folder_pairs = [args.eval_folders]
    
    main(args.input, args.train_folder1, args.train_folder2, eval_folder_pairs, args.epochs, args.batch_size, args.lr, args.output, args.model_path, args.load_model)