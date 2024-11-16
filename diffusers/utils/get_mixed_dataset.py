import os
import json
import shutil
import random
import argparse

def create_combined_dataset(a2, b2, a2_images, b2_images, a2_percentage, b2_percentage, dataset_name, 
                          a2_captions, b2_captions, previous_a2=None, previous_b2=None):
    """
    Create a combined dataset with specified percentages of A2 and B2 images.
    
    Args:
        a2_images (list): List of image paths from A2 dataset
        b2_images (list): List of image paths from B2 dataset
        a2_percentage (float): Percentage of A2 images to include (0-1)
        b2_percentage (float): Percentage of B2 images to include (0-1)
        dataset_name (str): Name of the output dataset
        a2_captions (dict): Captions for A2 images
        b2_captions (dict): Captions for B2 images
        previous_a2 (list): Previously selected A2 images to be included
        previous_b2 (list): Previously selected B2 images to be included
    
    Returns:
        tuple: Lists of selected A2 and B2 images
    """
    # Calculate number of images to select from each dataset
    num_a2 = int(len(a2_images) * a2_percentage)
    num_b2 = int(len(b2_images) * b2_percentage)
    
    # Initialize selected images with previous selections
    selected_a2 = previous_a2.copy() if previous_a2 is not None else []
    selected_b2 = previous_b2.copy() if previous_b2 is not None else []
    
    # Add additional random images if needed
    if len(selected_a2) < num_a2:
        remaining_a2 = list(set(a2_images) - set(selected_a2))
        additional_a2 = random.sample(remaining_a2, num_a2 - len(selected_a2))
        selected_a2.extend(additional_a2)
    
    if len(selected_b2) < num_b2:
        remaining_b2 = list(set(b2_images) - set(selected_b2))
        additional_b2 = random.sample(remaining_b2, num_b2 - len(selected_b2))
        selected_b2.extend(additional_b2)
    
    # Create dataset directory structure
    os.makedirs(f"{dataset_name}", exist_ok=True)
    os.makedirs(f"{dataset_name}/images", exist_ok=True)
    
    # Copy images and create caption dictionary
    captions = {}
    
    # Process A2 images with prefix
    for img in selected_a2:
        img_name = os.path.basename(img)
        base_name, ext = os.path.splitext(img_name)
        new_img_name = f"{a2}_{base_name}{ext}"
        shutil.copy2(img, f"{dataset_name}/images/{new_img_name}")
        captions[f"{a2}_{base_name}"] = {
            "path": new_img_name,
            "caption": a2_captions[base_name]["caption"],
            "width": a2_captions[base_name]["width"],
            "height": a2_captions[base_name]["height"]
        }
    
    # Process B2 images with prefix
    for img in selected_b2:
        img_name = os.path.basename(img)
        base_name, ext = os.path.splitext(img_name)
        new_img_name = f"{b2}_{base_name}{ext}"
        shutil.copy2(img, f"{dataset_name}/images/{new_img_name}")
        captions[f"{b2}_{base_name}"] = {
            "path": new_img_name,
            "caption": b2_captions[base_name]["caption"],
            "width": b2_captions[base_name]["width"],
            "height": b2_captions[base_name]["height"]
        }
    
    # Save caption.json
    with open(f"{dataset_name}/caption.json", 'w') as f:
        json.dump(captions, f, indent=4)
    
    return selected_a2, selected_b2

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create combined datasets from A2 and B2 images')
    parser.add_argument('--input_dir', default='datasets/', help='Root directory containing A2 and B2 directories')
    parser.add_argument('--output_dir', default='datasets/', help='Directory to store the combined datasets')
    parser.add_argument('--a2', required=True, help='Base name for the combined datasets')
    parser.add_argument('--b2', required=True, help='Base name for the combined datasets')
    args = parser.parse_args()

    # Define paths using arguments
    a2_path = os.path.join(args.input_dir, f"{args.a2}")
    b2_path = os.path.join(args.input_dir, f"{args.b2}")
    a2_images_path = os.path.join(a2_path, "images")
    b2_images_path = os.path.join(b2_path, "images")
    
    # Load captions
    with open(os.path.join(a2_path, "caption.json"), 'r') as f:
        a2_captions = json.load(f)
    with open(os.path.join(b2_path, "caption.json"), 'r') as f:
        b2_captions = json.load(f)
    
    # Get list of images
    a2_images = [os.path.join(a2_images_path, f) for f in os.listdir(a2_images_path) if f.endswith('.png')]
    b2_images = [os.path.join(b2_images_path, f) for f in os.listdir(b2_images_path) if f.endswith('.png')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets in order, maintaining proper nesting
    # First create 25% A2 + 75% B2
    selected_a2_25, selected_b2_75 = create_combined_dataset(
        args.a2, args.b2,
        a2_images, b2_images, 0.25, 0.75,
        os.path.join(args.output_dir, f"{args.a2}-25-{args.b2}-75"),
        a2_captions, b2_captions
    )
    
    # Create 50% A2 + 50% B2, including all previous selections
    selected_a2_50, selected_b2_50 = create_combined_dataset(
        args.a2, args.b2,
        a2_images, b2_images, 0.50, 0.50,
        os.path.join(args.output_dir, f"{args.a2}-50-{args.b2}-50"),
        a2_captions, b2_captions,
        selected_a2_25, selected_b2_75
    )
    
    # Create 75% A2 + 25% B2, including all previous selections
    create_combined_dataset(
        args.a2, args.b2,
        a2_images, b2_images, 0.75, 0.25,
        os.path.join(args.output_dir, f"{args.a2}-75-{args.b2}-25"),
        a2_captions, b2_captions,
        selected_a2_50, selected_b2_50
    )

if __name__ == "__main__":
    main()