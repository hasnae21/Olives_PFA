from ultralytics import YOLO
import torch
import os

def train():
    # 1. Load the model
    model = YOLO('yolov8m.pt')

    # 2. Resolve data.yaml path relative to this script (avoids working-directory issues)
    data_path = r"C:/Users/louay/Downloads/olives_2/dataset/augmented_split/data.yaml"

    # 3. Pick device: use CUDA if available, fall back to CPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 4. Launch training
    model.train(
        data=data_path,
        epochs=100,
        rect=False,       # Use rectangular training (faster on non-square images)
        imgsz=1280,
        device=device,   # RTX 3060 GPU (falls back to CPU if unavailable)
        workers=2,       # Keep low on Windows to avoid multiprocessing issues
        batch=16,
        patience=20      # Increased from 10: avoids premature early stopping during LR plateaus
    )

# Required on Windows for multiprocessing safety
if __name__ == '__main__':
    train()
