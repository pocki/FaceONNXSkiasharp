# FaceONNX 👤

Lightweight .NET 10 face detection library based on ONNX Runtime and SkiaSharp.

## Projects 📦

- `FaceONNX`: core library
- `FaceONNX.Samples`: console sample app (`images/` -> `results/`)

## Quick start 🚀

```bash
cd FaceONNX.Samples
dotnet run
```

## Usage 🧩

```csharp
using FaceONNX;
using SkiaSharp;

using var detector = new FaceDetector(model: FaceDetectorModel.Yolov5);
using var bitmap = SKBitmap.Decode("images\\group.jpg");

FaceDetectionResult[] detections = detector.ForwardDetection(bitmap);
SKRectI[] boxes = detector.Forward(bitmap);

var roi = new SKRectI(100, 80, 500, 420);
FaceDetectionResult[] roiDetections = detector.ForwardDetection(bitmap, roi, clamp: true);
```

## Model selection 🧠

```csharp
// Default
using var yolov5Detector = new FaceDetector(model: FaceDetectorModel.Yolov5);

using var yolo26Detector = new FaceDetector(model: FaceDetectorModel.Yolo26);
using var faceOnnxDetector = new FaceDetector(model: FaceDetectorModel.FaceOnnx);
```

| Model enum | ONNX file | Model size | Output | Landmarks |
|---|---|---|---|---|
| `FaceDetectorModel.Yolov5` | `yolov5s-face.onnx` | 29.3 MB | Single YOLOv5-style tensor | Yes (5-point) |
| `FaceDetectorModel.Yolo26` | `yolo26_face_fp16.onnx` | 18.3 MB | Single YOLO tensor (`xyxy + confidence + class`) | No |
| `FaceDetectorModel.FaceOnnx` | `face_detector_640.onnx` | 1.5 MB | Split outputs (`confidences + boxes`) | No |

## API summary 🛠️

- `Forward(...)` returns `SKRectI[]`
- `ForwardDetection(...)` returns `FaceDetectionResult[]`
- ROI overloads: `Forward(image, rectangle, clamp)` and `ForwardDetection(image, rectangle, clamp)`
- Raw float-channel overloads: `Forward(float[][,])` and `ForwardDetection(float[][,])` (CHW, 0–255 range)
- Thresholds: `detectionThreshold`, `confidenceThreshold`, `nmsThreshold`
- Default model: `FaceDetectorModel.Yolov5`

## Attribution and licenses 🙏

### FaceONNX model (`face_detector_640.onnx`) 🧾

Source:

- <https://github.com/arieffauzi-st/FaceONNX>
- <https://github.com/arieffauzi-st/FaceONNX/blob/main/FaceONNX/Models/face_detector_640.onnx>

Thanks to `arieffauzi-st` for maintaining and sharing this FaceONNX fork.

License status in source repo:

- no root `LICENSE` file
- no model-specific license/notice file in `FaceONNX/Models`

Reference:

- <https://github.com/arieffauzi-st/FaceONNX/blob/main/README.md>

### YOLOv5 model and related integration code 🤖

Source:

- <https://github.com/FaceONNX/FaceONNX>
- <https://github.com/FaceONNX/FaceONNX.Models>

Thanks to the FaceONNX maintainers and contributors for publishing the models and implementation details used by this project.

License:

- MIT
- Copyright (c) 2020-2025 Valery Asiryan
- base models: Ultralytics YOLOv5 (AGPL-3.0, or Ultralytics Enterprise License)

Reference:

- <https://github.com/FaceONNX/FaceONNX/blob/main/LICENSE>
- <https://github.com/FaceONNX/FaceONNX.Models/blob/main/LICENSE>

### YOLO26 model (`yolo26_face_fp16.onnx`) 🆕

Source:

- <https://github.com/marceloeatworld/yolo26-training>

License details from source repo:

- training code: MIT
- base models: Ultralytics YOLO26 (AGPL-3.0, or Ultralytics Enterprise License)
- dataset terms: WiderFace terms

Reference:

- <https://github.com/marceloeatworld/yolo26-training/blob/main/README.md>
