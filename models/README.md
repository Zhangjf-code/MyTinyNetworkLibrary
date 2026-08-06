# YOLOv8n inference models

This directory contains artifacts used by the TensorFlow CPU and TensorRT GPU backends.

## Files

- `yolov8n.pt`: Ultralytics YOLOv8n detection weights, downloaded from the official Ultralytics v8.3.0 assets release.
- `yolov8n_tf218_saved_model/`: TensorFlow 2.18 SavedModel generated from those weights.
- `yolov8n_tf218_saved_model/signature.json`: stable serving signature consumed by the C++ backend.
- `yolov8n.onnx`: fixed `[1,3,640,640]` ONNX exported with opset 12.
- `yolov8n_fp16.engine`: TensorRT 10.0.1 FP16 engine built on the RTX 4060 Ti in this development machine.

SHA-256:

```text
f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36  yolov8n.pt
505efb624ba08a2546f35d660d831cb11c5f2a70c323fa53699ec9fd9ba4a775  yolov8n_tf218_saved_model/saved_model.pb
67c0194d883cb8a1e86a5f8dc4af7411478a2a7f9cc12468d673e3068e96c834  yolov8n.onnx
ac22781558527900e7b571b15252249a747197f1b23fe4231609ff46e1d4ba6d  yolov8n_fp16.engine
```

The export can be reproduced with `tools/export_yolov8_savedmodel.py` in an environment containing Ultralytics 8.3.0 and TensorFlow 2.18.0.

The TensorRT artifacts can be reproduced with `tools/export_yolov8_onnx.py` and `tools/build_tensorrt_engine.sh`. TensorRT engines are not generally portable across TensorRT versions or GPU architectures; rebuild the engine on the deployment target when either changes.

Ultralytics software and model assets have their own license terms. Review the current [Ultralytics licensing documentation](https://www.ultralytics.com/license) before distributing or using the model commercially.
