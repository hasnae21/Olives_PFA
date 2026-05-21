from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
from collections import Counter


def compute_class_weights(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base_dir = Path(data_yaml_path).parent
    train_labels_dir = base_dir / 'train' / 'labels'

    class_counts = Counter()
    for label_file in train_labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1

    num_classes = len(data['names'])
    total = sum(class_counts.values())

    print("Distribution des classes :")
    for i, name in enumerate(data['names']):
        count = class_counts.get(i, 0)
        print(f"  Classe {i} ({name}): {count} instances ({100*count/total:.1f}%)")

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weights.append(total / (num_classes * count))

    min_w = min(weights)
    weights = [round(w / min_w, 3) for w in weights]

    print(f"Poids calculés : {weights}")
    return weights


def train():
    model = YOLO('yolov8m.pt')

    data_path = r"C:/Users/louay/Downloads/olives_2/dataset/split/data.yaml"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cls_weights = compute_class_weights(data_path)
    weights_tensor = torch.tensor(cls_weights, dtype=torch.float)

    # Callback : injecte les poids dans la loss de classification après init du trainer
    def on_train_start(trainer):
        if hasattr(trainer, 'criterion') and hasattr(trainer.criterion, 'cls'):
            trainer.criterion.cls.weight = weights_tensor.to(trainer.device)
            print(f"Poids de classes injectés : {cls_weights}")

    model.add_callback('on_train_start', on_train_start)

    model.train(
        data=data_path,
        epochs=100,
        imgsz=1240,
        device=device,
        workers=2,
        batch=16,
        patience=20,
        rect=False,
        cls=1.5,
        copy_paste=0.1,
        mosaic=1.0,
        mixup=0.1,
    )


if __name__ == '__main__':
    train()