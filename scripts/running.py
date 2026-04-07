import argparse
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
from ultralytics import YOLO


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

		p1 = (int(x1), int(y1))
		p2 = (int(x2), int(y2))
		cv2.rectangle(output, p1, p2, (0, 200, 0), 2)

		(text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
		text_y = max(text_h + 6, p1[1])
		cv2.rectangle(
			output,
			(p1[0], text_y - text_h - baseline - 6),
			(p1[0] + text_w + 8, text_y + 2),
			(0, 200, 0),
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run olive detection on one image and display the annotated result."
	)
	parser.add_argument(
		"--image",
		type=str,
		default=None,
		help="Path to input image. If omitted, a file picker will open.",
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

	if args.image:
		image_path = Path(args.image).resolve()
	else:
		image_path = choose_image_path()

	if not model_path.exists():
		raise FileNotFoundError(f"Model not found: {model_path}")
	if not image_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")

	original = cv2.imread(str(image_path))
	if original is None:
		raise RuntimeError(f"Unable to read image: {image_path}")
	h, w = original.shape[:2]

	model = YOLO(str(model_path))
	results = model.predict(
		source=str(image_path),
		conf=args.conf,
		imgsz=args.imgsz,
		verbose=False,
	)

	if not results:
		raise RuntimeError("No inference result returned by YOLO.")

	annotated = draw_detections(original, results[0])
	display_image, display_scale = fit_image_to_screen(annotated)

	window_name = "Olive Detection"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(window_name, display_image.shape[1], display_image.shape[0])
	cv2.imshow(window_name, display_image)
	print(f"Original image size: {w}x{h}")
	print(f"Inference imgsz: {args.imgsz}")
	if display_scale < 1.0:
		print(f"Display scale applied: {display_scale:.3f} (full image fitted to screen)")
	else:
		print("Display scale applied: 1.000 (no downscale)")
	n_det = len(results[0].boxes) if results[0].boxes is not None else 0
	print(f"Detected objects: {n_det}")
	if n_det:
		confs = results[0].boxes.conf.tolist()
		print(f"Confidences: {[round(c, 3) for c in confs]}")
	else:
		print("No objects detected. Try lowering --conf (e.g. 0.05).")
	print("Press any key in the image window to close.")
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()