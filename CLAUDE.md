# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Facial landmark detection using YOLOv3/Darknet. Each of 5 facial landmarks (left_eye, right_eye, nose_tip, left_mouth, right_mouth) is treated as a micro-object (~2.5% of image size bounding box) for standard YOLO object detection. The project is written in Chinese context (comments, docs).

## Architecture

```
300-W Dataset (.pts) → convert_300w_to_yolo.py → YOLO labels (.txt)
                                                       ↓
                                          generate_train_val.py → train.txt / val.txt
                                                       ↓
                                          Darknet C executable trains YOLOv3
                                                       ↓
                                          .weights file → inference.py (OpenCV DNN + webcam)
```

**Key design decision**: Landmarks are encoded as tiny bounding boxes (class_id, x_center, y_center, 0.025, 0.025) in YOLO format, not as regression keypoints. This allows using unmodified Darknet.

## Commands

### Data Preparation
```bash
# Convert 300-W dataset to YOLO format (extracts 5 points from 68-point annotations)
python scripts/convert_300w_to_yolo.py --input_dir <300W_PATH> --output_images data/images/train --output_labels data/labels/train

# Generate 80/20 train/val split (moves val files, creates path lists)
python scripts/generate_train_val.py
```

### Training (requires compiled Darknet executable)
```bash
# Initial training with pretrained backbone
./darknet detector train cfg/face_landmark.data cfg/yolov3-face-landmark.cfg darknet53.conv.74

# Continue training from checkpoint
./darknet detector train cfg/face_landmark.data cfg/yolov3-face-landmark.cfg weights/yolov3-face-landmark_last.weights
```

### Inference
```bash
python inference.py                          # default webcam
python inference.py --camera 1 --conf 0.3    # alternate camera, lower threshold
```

### Manual Annotation (for adding custom training data)
```bash
python tools/landmark_annotator.py --input_dir <IMAGES> --output_dir <LABELS>
# Click 5 points per image: left_eye → right_eye → nose → left_mouth → right_mouth
# Keys: r=reset, n=skip, s=save, q=quit
```

## YOLOv3 Config Conventions

In `cfg/yolov3-face-landmark.cfg`, when changing the number of classes:
- Update `classes=N` in all 3 `[yolo]` sections
- Update `filters=(N+5)*3` in the `[convolutional]` layer immediately before each `[yolo]` section
- Current: classes=5, filters=30

## Landmark Class Mapping (from 300-W 68-point format)

| Class ID | Name | 300-W Indices |
|----------|------|---------------|
| 0 | left_eye | avg of [36-41] |
| 1 | right_eye | avg of [42-47] |
| 2 | nose_tip | [30] |
| 3 | left_mouth | [48] |
| 4 | right_mouth | [54] |

## Dependencies

Only `opencv-python` and `numpy` are required. No requirements.txt exists — install with `pip install opencv-python numpy`.

## Known Limitations

- YOLO struggles with micro-objects; the 2.5% bbox size (`BBOX_RELATIVE_SIZE = 0.025`) may need tuning
- Training requires the Darknet C executable (not Python); AlexeyAB/darknet fork recommended for Windows
- Python is used only for data prep, annotation, and inference (via `cv2.dnn`)
