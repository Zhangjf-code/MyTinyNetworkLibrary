#!/usr/bin/env python3
"""Export YOLOv8 weights and add a stable TensorFlow serving signature."""

import argparse
import json
from pathlib import Path

import tensorflow as tf
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--output", default="yolov8n_tf_saved_model")
    parser.add_argument("--raw-saved-model", help="Wrap an existing Ultralytics SavedModel instead of exporting")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.raw_saved_model:
        exported = Path(args.raw_saved_model).resolve()
    else:
        weights = Path(args.weights).resolve()
        expected_export = weights.with_name(weights.stem + "_saved_model")
        try:
            exported = Path(YOLO(str(weights)).export(
                format="saved_model", imgsz=args.imgsz, batch=1, device="cpu"
            ))
        except Exception:
            # Ultralytics 8.3 may fail while attaching optional TFLite USB
            # metadata after it has already written a valid SavedModel.
            if not (expected_export / "saved_model.pb").is_file():
                raise
            exported = expected_export
    raw_model = tf.saved_model.load(str(exported))

    class ServingModule(tf.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=[1, args.imgsz, args.imgsz, 3],
                    dtype=tf.float32,
                    name="images",
                )
            ]
        )
        def serve(self, images):
            return {"output0": self.model(images)}

    output = Path(args.output).resolve()
    module = ServingModule(raw_model)
    tf.saved_model.save(
        module,
        str(output),
        signatures={"serving_default": module.serve.get_concrete_function()},
    )
    signature = {
        "signature": "serving_default",
        "input": {"key": "images", "shape": [1, args.imgsz, args.imgsz, 3], "dtype": "float32"},
        "output": {"key": "output0", "shape": [1, 84, 8400], "dtype": "float32"},
        "layout": "NHWC",
        "normalization": "RGB float32 / 255.0",
    }
    (output / "signature.json").write_text(
        json.dumps(signature, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
