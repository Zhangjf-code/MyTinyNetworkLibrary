#!/usr/bin/env python3
"""Export fixed-shape YOLOv8 detection weights for TensorRT."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()
    exported = YOLO(str(Path(args.weights).resolve())).export(
        format="onnx",
        imgsz=args.imgsz,
        batch=1,
        dynamic=False,
        simplify=True,
        opset=args.opset,
        device="cpu",
    )
    print(exported)


if __name__ == "__main__":
    main()
