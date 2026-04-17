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


def get_screen_bounds() -> tuple[int, int]:
	"""Get max display bounds based on current screen size."""
	tmp = Tk()
	tmp.withdraw()
	screen_w = tmp.winfo_screenwidth()
	screen_h = tmp.winfo_screenheight()
	tmp.destroy()
	return int(screen_w * 0.95), int(screen_h * 0.9)


def fit_image_to_bounds(image, max_w: int, max_h: int):
	"""Resize only for display so frame fits in the screen bounds."""
	h, w = image.shape[:2]
	scale = min(max_w / w, max_h / h, 1.0)
	new_w = max(1, int(w * scale))
	new_h = max(1, int(h * scale))

	if scale < 1.0:
		resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
		return resized, scale
	return image, 1.0


def draw_detections(image, result):
	"""Draw rectangles and class labels on a video frame."""
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


def choose_video_paths() -> list[Path]:
	"""Open a file picker and return selected video paths."""
	root = Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	selected = filedialog.askopenfilenames(
		title="Select one or more videos",
		filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v")],
	)
	root.destroy()
	if not selected:
		raise SystemExit("No videos selected. Exiting.")
	return [Path(p) for p in selected]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run olive detection on one or more videos and display annotated frames."
	)
	parser.add_argument(
		"--videos",
		type=str,
		nargs="+",
		default=None,
		help="One or more video paths. If omitted, a multi-file picker will open.",
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
	parser.add_argument(
		"--delay-ms",
		type=int,
		default=30,
		help="Extra delay in milliseconds between displayed frames (higher = slower playback).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	project_root = Path(__file__).resolve().parent.parent
	model_path = (project_root / args.model).resolve()

	if args.videos:
		video_paths = [Path(p).resolve() for p in args.videos]
	else:
		video_paths = [p.resolve() for p in choose_video_paths()]

	if not model_path.exists():
		raise FileNotFoundError(f"Model not found: {model_path}")
	for video_path in video_paths:
		if not video_path.exists():
			raise FileNotFoundError(f"Video not found: {video_path}")

	model = YOLO(str(model_path))
	max_w, max_h = get_screen_bounds()
	delay_ms = max(1, args.delay_ms)

	overall_counts = Counter()
	overall_detections = 0
	overall_frames = 0
	stop_all = False

	for vid_idx, video_path in enumerate(video_paths, start=1):
		cap = cv2.VideoCapture(str(video_path))
		if not cap.isOpened():
			raise RuntimeError(f"Unable to open video: {video_path}")

		video_counts = Counter()
		video_detections = 0
		frame_idx = 0

		window_name = f"Olive Detection Video {vid_idx}/{len(video_paths)}"
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

		print("=" * 60)
		print(f"Processing video {vid_idx}/{len(video_paths)}: {video_path}")
		print(f"Inference imgsz: {args.imgsz}")
		print(f"Display delay: {delay_ms} ms per frame")
		print("Keyboard: press 'n' for next video, 'q' to quit all.")

		while True:
			ok, frame = cap.read()
			if not ok:
				break

			frame_idx += 1
			overall_frames += 1

			results = model.predict(
				source=frame,
				conf=args.conf,
				imgsz=args.imgsz,
				verbose=False,
			)
			if not results:
				continue

			result = results[0]
			annotated = draw_detections(frame, result)

			frame_counts = Counter()
			boxes = result.boxes
			names = result.names
			if boxes is not None and len(boxes) > 0:
				for cls_tensor in boxes.cls:
					cls_id = int(cls_tensor.item())
					label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
					frame_counts[label] += 1

			n_det = sum(frame_counts.values())
			video_detections += n_det
			overall_detections += n_det
			video_counts.update(frame_counts)
			overall_counts.update(frame_counts)

			status = f"Frame: {frame_idx} | Detections: {n_det} | Delay: {delay_ms}ms | q: quit | n: next"
			cv2.putText(
				annotated,
				status,
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.75,
				(255, 255, 255),
				2,
			)

			display_frame, _ = fit_image_to_bounds(annotated, max_w, max_h)
			cv2.resizeWindow(window_name, display_frame.shape[1], display_frame.shape[0])
			cv2.imshow(window_name, display_frame)

			key = cv2.waitKey(delay_ms) & 0xFF
			if key == ord("n"):
				break
			if key == ord("q"):
				stop_all = True
				break

		cap.release()
		cv2.destroyWindow(window_name)

		print(f"Frames processed in this video: {frame_idx}")
		print(f"Detected objects in this video: {video_detections}")
		if video_counts:
			print("Category counts in this video:")
			for label, count in sorted(video_counts.items()):
				print(f"  - {label}: {count}")
		else:
			print("No objects detected in this video.")

		if stop_all:
			break

	cv2.destroyAllWindows()

	print("=" * 60)
	print("Overall summary (all selected videos):")
	print(f"Total videos selected: {len(video_paths)}")
	print(f"Total frames processed: {overall_frames}")
	print(f"Total detected objects: {overall_detections}")
	if overall_counts:
		print("Total category counts:")
		for label, count in sorted(overall_counts.items()):
			print(f"  - {label}: {count}")


if __name__ == "__main__":
	main()
