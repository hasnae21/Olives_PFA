import argparse
from collections import Counter
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
from ultralytics import YOLO


CLASS_COLORS = [
	(0, 200, 0),
	(255, 170, 0),
	(0, 120, 255),
	(220, 80, 80),
	(180, 0, 180),
	(0, 200, 200),
	(255, 80, 180),
	(80, 80, 255),
]


def get_class_color(cls_id: int) -> tuple[int, int, int]:
	"""Return a stable BGR color for a class id."""
	return CLASS_COLORS[cls_id % len(CLASS_COLORS)]


def draw_detections(image, result):
	"""Draw rectangles and class labels on the original image."""
	output = image.copy()
	boxes = result.boxes
	names = result.names

	if boxes is None:
		return output

	for box in boxes:
		x1, y1, x2, y2 = box.xyxy[0].tolist()
		cls_id = int(box.cls[0].item())
		conf = float(box.conf[0].item())
		label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
		text = f"{label} {conf:.2f}"
		color = get_class_color(cls_id)

		p1 = (int(x1), int(y1))
		p2 = (int(x2), int(y2))
		cv2.rectangle(output, p1, p2, color, 2)

		(text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
		text_y = max(text_h + 6, p1[1])
		cv2.rectangle(
			output,
			(p1[0], text_y - text_h - baseline - 6),
			(p1[0] + text_w + 8, text_y + 2),
			color,
			-1,
		)
		cv2.putText(output, text, (p1[0] + 4, text_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

	return output


def fit_image_to_screen(image):
	"""Resize only for display so the full image fits on screen."""
	h, w = image.shape[:2]
	tmp = Tk()
	tmp.withdraw()
	screen_w = tmp.winfo_screenwidth()
	screen_h = tmp.winfo_screenheight()
	tmp.destroy()

	max_w = int(screen_w * 0.95)
	max_h = int(screen_h * 0.9)
	scale = min(max_w / w, max_h / h, 1.0)
	new_w = max(1, int(w * scale))
	new_h = max(1, int(h * scale))

	if scale < 1.0:
		resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
		return resized, scale
	return image, 1.0


def choose_image_path() -> Path:
	"""Open a file picker and return the selected image path."""
	root = Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	selected = filedialog.askopenfilename(
		title="Select an image",
		filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")],
	)
	root.destroy()
	if not selected:
		raise SystemExit("No image selected. Exiting.")
	return Path(selected)


def choose_image_paths() -> list[Path]:
	"""Open a file picker and return selected image paths."""
	root = Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	selected = filedialog.askopenfilenames(
		title="Select one or more images",
		filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")],
	)
	root.destroy()
	if not selected:
		raise SystemExit("No images selected. Exiting.")
	return [Path(p) for p in selected]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run olive detection on one or more images and display annotated results."
	)
	parser.add_argument(
		"--images",
		type=str,
		nargs="+",
		default=None,
		help="One or more image paths. If omitted, a multi-file picker will open.",
	)
	parser.add_argument(
		"--model",
		type=str,
		default="runs/detect/train3/weights/best.pt",
		help="Path to trained YOLO model weights.",
	)
	parser.add_argument(
		"--conf",
		type=float,
		default=0.15,
		help="Confidence threshold.",
	)
	parser.add_argument(
		"--imgsz",
		type=int,
		default=1280,
		help="Inference image size (recommended: same as training, e.g. 1280).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	project_root = Path(__file__).resolve().parent.parent
	model_path = (project_root / args.model).resolve()

	if args.images:
		image_paths = [Path(p).resolve() for p in args.images]
	else:
		image_paths = [p.resolve() for p in choose_image_paths()]

	if not model_path.exists():
		raise FileNotFoundError(f"Model not found: {model_path}")
	for image_path in image_paths:
		if not image_path.exists():
			raise FileNotFoundError(f"Image not found: {image_path}")

	model = YOLO(str(model_path))

	global_counts = Counter()
	total_detections = 0

	for idx, image_path in enumerate(image_paths, start=1):
		original = cv2.imread(str(image_path))
		if original is None:
			raise RuntimeError(f"Unable to read image: {image_path}")
		h, w = original.shape[:2]

		results = model.predict(
			source=str(image_path),
			conf=args.conf,
			imgsz=args.imgsz,
			verbose=False,
		)

		if not results:
			raise RuntimeError(f"No inference result returned by YOLO for: {image_path}")

		result = results[0]
		annotated = draw_detections(original, result)
		display_image, display_scale = fit_image_to_screen(annotated)

		window_name = "Olive Detection"
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(window_name, display_image.shape[1], display_image.shape[0])
		cv2.imshow(window_name, display_image)

		image_counts = Counter()
		boxes = result.boxes
		names = result.names
		if boxes is not None and len(boxes) > 0:
			for cls_tensor in boxes.cls:
				cls_id = int(cls_tensor.item())
				label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
				image_counts[label] += 1

		n_det = sum(image_counts.values())
		total_detections += n_det
		global_counts.update(image_counts)

		print("=" * 60)
		print(f"Image {idx}/{len(image_paths)}: {image_path}")
		print(f"Original image size: {w}x{h}")
		print(f"Inference imgsz: {args.imgsz}")
		if display_scale < 1.0:
			print(f"Display scale applied: {display_scale:.3f} (full image fitted to screen)")
		else:
			print("Display scale applied: 1.000 (no downscale)")
		print(f"Detected objects in this image: {n_det}")
		if image_counts:
			print("Category counts in this image:")
			for label, count in sorted(image_counts.items()):
				print(f"  - {label}: {count}")
			confs = result.boxes.conf.tolist()
			print(f"Confidences: {[round(c, 3) for c in confs]}")
		else:
			print("No objects detected in this image. Try lowering --conf (e.g. 0.05).")

		if idx < len(image_paths):
			print("Press any key in the image window for the next image.")
		else:
			print("Press any key in the image window to finish.")
		cv2.waitKey(0)

	cv2.destroyAllWindows()

	print("=" * 60)
	print("Overall summary (all selected images):")
	print(f"Total images processed: {len(image_paths)}")
	print(f"Total detected objects: {total_detections}")
	if global_counts:
		print("Total category counts:")
		for label, count in sorted(global_counts.items()):
			print(f"  - {label}: {count}")


if __name__ == "__main__":
	main()