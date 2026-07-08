# Face Detection and Multiple Face Recognition with FaceOnnx 👤

A lightweight .NET 10 face detection library built on ONNX Runtime + SkiaSharp.

## Why this project? ✨

- ✅ Removed heavy imaging dependencies: `Emgu.CV`, `OpenCvSharp4`, `UMapx`, `System.Drawing`
- ✅ Uses `SkiaSharp` for image processing and rendering
- ✅ Targets `.NET 10`

## Projects

- `FaceONNX`: core library (face detection and processing)
- `FaceONNX.Samples`: console sample app with `images/` and `results/`

## Run the sample 🚀

```bash
cd FaceONNX.Samples
dotnet run
dotnet run -- --test
```

## Library usage 🧩

### 1) Basic detection from an image

```csharp
using FaceONNX;
using SkiaSharp;

using var detector = new FaceDetector(model: FaceDetectorModel.Yolo);
using var bitmap = SKBitmap.Decode("images\\group.jpg");

FaceDetectionResult[] detections = detector.ForwardDetection(bitmap);

foreach (var d in detections)
{
    Console.WriteLine($"Score={d.Score:F3}, Box={d.Rectangle}");
}
```

### 2) Get rectangles only

```csharp
SKRectI[] boxes = detector.Forward(bitmap);
```

### 3) Detect inside a ROI (region of interest)

```csharp
var roi = new SKRectI(100, 80, 500, 420);
FaceDetectionResult[] roiDetections = detector.ForwardDetection(bitmap, roi, clamp: true);
```

## Model selection 🧠

`FaceDetector` now supports explicit model selection via `FaceDetectorModel`:

```csharp
// YOLO model (default)
using var yoloDetector = new FaceDetector(model: FaceDetectorModel.Yolo);

// FaceONNX split-output model
using var faceOnnxDetector = new FaceDetector(model: FaceDetectorModel.FaceOnnx);
```

## Model outputs explained 📦

| Model | ONNX file | Output layout | Landmarks | Notes |
|---|---|---|---|---|
| 🤖 `FaceDetectorModel.Yolo` | `yolov5s-face.onnx` | Single YOLO-style tensor | ✅ 5-point landmarks | Good when you need face keypoints (eyes, nose, mouth corners) |
| 🧾 `FaceDetectorModel.FaceOnnx` | `face_detector_640.onnx` | Split outputs (confidences + boxes) | ❌ No landmarks | Good when you only need fast face boxes |

### Practical difference

- 🎯 **YOLO** returns `Rectangle` + `Score` + `Points` (landmarks)
- 📌 **FaceOnnx** returns `Rectangle` + `Score` (no landmarks)

## API notes 🛠️

- `Forward(...)` returns only `SKRectI[]` boxes
- `ForwardDetection(...)` returns rich `FaceDetectionResult[]`
- Thresholds are configurable via constructor (`detectionThreshold`, `confidenceThreshold`, `nmsThreshold`)
- Default model is `FaceDetectorModel.Yolo`

## Extensions and helper methods 🔧

The library includes convenient extension helpers for post-processing:

- 🖼️ `FaceProcessingExtensions.Align(...)`: rotate/align full image or face ROI (`SKBitmap` and RGB channel-array overloads)
- 📐 `Rectangles.ToBox(...)`, `Rectangles.Scale(...)`: convert rectangles to square boxes and expand/shrink regions
- 🎯 `Rectangles.IoU(...)`: compute overlap ratio (Intersection over Union)
- 👀 `Face5Landmarks`: helpers like `LeftEye`, `RightEye`, `Nose`, `Mouth`, plus `RotationAngle` and `SymmetryCoefficient`

### Short sample

```csharp
using FaceONNX;
using SkiaSharp;

using var detector = new FaceDetector(model: FaceDetectorModel.Yolo);
using var bitmap = SKBitmap.Decode("images\\group.jpg");

var detections = detector.ForwardDetection(bitmap);
if (detections.Length == 0)
{
    return;
}

var face = detections[0];

// Expand to a square crop around the first face.
SKRectI square = face.Rectangle.ToBox(0.25f);

// If landmarks are present (YOLO model), align by estimated face roll.
if (face.Points is not null)
{
    using var aligned = bitmap.Align(square, face.Points.RotationAngle, clamp: true);
    Console.WriteLine($"Roll angle = {face.Points.RotationAngle:F2}°");
}

Console.WriteLine($"IoU(original, square) = {face.Rectangle.IoU(square):F3}");
```

