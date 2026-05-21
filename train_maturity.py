"""
CNN Training Script for Olive Maturity Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# Import our Maturity CNN model
from maturity_classifier.cnn_model import MaturityCNN


class MaturityDataset(Dataset):
    """Dataset for maturity classification task"""
    def __init__(self, images_dir, labels_source, image_size=224, num_classes=3):
        """
        Args:
            images_dir (str): Directory containing olive region images
            labels_source (str): Either a text file with labels (one per line) or a directory of per-image label files
            image_size (int): Size to resize images to
            num_classes (int): Number of maturity classes
        """
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.num_classes = num_classes

        # Get list of images
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.JPG'))])

        self.labels_source = Path(labels_source)
        self.label_mode = "directory" if self.labels_source.is_dir() else "file"

        if self.label_mode == "file":
            self.labels = []
            with open(self.labels_source, 'r', encoding='utf-8') as f:
                self.labels = [int(line.strip()) for line in f.readlines() if line.strip()]

            # Ensure we have labels for all images
            assert len(self.labels) == len(self.image_files), \
                f"Number of labels ({len(self.labels)}) != number of images ({len(self.image_files)})"
        else:
            self.labels = None

    @staticmethod
    def _read_single_label_file(label_path: Path) -> int:
        lines = [line.strip() for line in label_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Empty label file: {label_path}")

        # Accept either a single integer line or a YOLO line, extracting the class id from the first value.
        return int(float(lines[0].split()[0]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images_dir / self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        # Get label
        if self.label_mode == "file":
            label_value = self.labels[idx]
        else:
            label_path = self.labels_source / f"{img_path.stem}.txt"
            label_value = self._read_single_label_file(label_path)

        label = torch.tensor(label_value, dtype=torch.long)

        return image, label


def train_maturity_classifier(
    images_dir,
    labels_source,
    output_dir="models",
    batch_size=32,
    epochs=50,
    learning_rate=1e-3,
    image_size=224,
    num_classes=3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train Maturity CNN model
    
    Args:
        images_dir (str): Directory with olive region images
        labels_source (str): File with labels or directory containing per-image label files
        output_dir (str): Directory to save model
        batch_size (int): Batch size
        epochs (int): Number of epochs
        learning_rate (float): Learning rate
        image_size (int): Image size
        num_classes (int): Number of maturity classes
        device (str): Device to use
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training on device: {device}")
    
    # Create model
    model = MaturityCNN(num_classes=num_classes, input_size=image_size)
    model = model.to(device)
    
    # Create dataset and dataloader
    dataset = MaturityDataset(images_dir, labels_source, image_size=image_size, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                "loss": loss.item(),
                "acc": 100 * correct / total
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(output_dir, "maturity_cnn_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Best model saved: {checkpoint_path}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(output_dir, f"maturity_cnn_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
        
        scheduler.step()
    
    print("Training completed!")
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN for olive maturity classification")
    parser.add_argument("--images", type=str, required=True, help="Path to training images")
    parser.add_argument("--labels", type=str, help="Path to labels file")
    parser.add_argument("--labels-dir", type=str, help="Path to directory containing per-image label files")
    parser.add_argument("--output", type=str, default="models", help="Output directory for models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of maturity classes")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    labels_source = args.labels_dir or args.labels
    if not labels_source:
        raise SystemExit("Provide either --labels or --labels-dir")

    train_maturity_classifier(
        images_dir=args.images,
        labels_source=labels_source,
        output_dir=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        image_size=args.image_size,
        num_classes=args.num_classes,
        device=device
    )
