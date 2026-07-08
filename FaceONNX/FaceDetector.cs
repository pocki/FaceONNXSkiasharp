using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace FaceONNX;

/// <summary>
/// Face detector model.
/// </summary>
public enum FaceDetectorModel
{
    /// <summary>
    /// FaceONNX split-output model (<c>face_detector_640.onnx</c>).
    /// </summary>
    FaceOnnx,

    /// <summary>
    /// YOLOv5 single-output model (<c>yolov5s-face.onnx</c>).
    /// </summary>
    Yolov5,

    /// <summary>
    /// YOLO26 single-output model (<c>yolo26_face_fp16.onnx</c>).
    /// </summary>
    Yolo26
}

/// <summary>
/// Defines face detector.
/// </summary>
public class FaceDetector : IFaceDetector
{
    /// <summary>
    /// Inference session.
    /// </summary>
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int _inputWidth;
    private readonly int _inputHeight;
    private readonly bool _yoloOutputMode;

    /// <summary>
    /// Initializes face detector.
    /// </summary>
    /// <param name="confidenceThreshold">Confidence threshold</param>
    /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
    /// <param name="model">Model variant to use</param>
    public FaceDetector(float confidenceThreshold = 0.4f, float nmsThreshold = 0.5f, FaceDetectorModel model = FaceDetectorModel.Yolov5)
        : this(0.3f, confidenceThreshold, nmsThreshold, model)
    {
    }

    /// <summary>
    /// Initializes face detector.
    /// </summary>
    /// <param name="detectionThreshold">Detection threshold</param>
    /// <param name="confidenceThreshold">Confidence threshold</param>
    /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
    /// <param name="model">Model variant to use</param>
    public FaceDetector(float detectionThreshold, float confidenceThreshold, float nmsThreshold, FaceDetectorModel model = FaceDetectorModel.Yolov5)
    {
        _session = new InferenceSession(ResolveModelPath(GetModelFileName(model)));
        (_inputName, _inputWidth, _inputHeight) = ReadInputInfo(_session);
        _yoloOutputMode = IsYoloOutputModel(_session);
        DetectionThreshold = detectionThreshold;
        ConfidenceThreshold = confidenceThreshold;
        NmsThreshold = nmsThreshold;
    }

    /// <summary>
    /// Initializes face detector.
    /// </summary>
    /// <param name="options">Session options</param>
    /// <param name="confidenceThreshold">Confidence threshold</param>
    /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
    /// <param name="model">Model variant to use</param>
    public FaceDetector(SessionOptions options, float confidenceThreshold = 0.4f, float nmsThreshold = 0.5f, FaceDetectorModel model = FaceDetectorModel.Yolov5)
        : this(options, 0.3f, confidenceThreshold, nmsThreshold, model)
    {
    }

    /// <summary>
    /// Initializes face detector.
    /// </summary>
    /// <param name="options">Session options</param>
    /// <param name="detectionThreshold">Detection threshold</param>
    /// <param name="confidenceThreshold">Confidence threshold</param>
    /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
    /// <param name="model">Model variant to use</param>
    public FaceDetector(SessionOptions options, float detectionThreshold, float confidenceThreshold, float nmsThreshold, FaceDetectorModel model = FaceDetectorModel.Yolov5)
    {
        _session = new InferenceSession(ResolveModelPath(GetModelFileName(model)), options);
        (_inputName, _inputWidth, _inputHeight) = ReadInputInfo(_session);
        _yoloOutputMode = IsYoloOutputModel(_session);
        DetectionThreshold = detectionThreshold;
        ConfidenceThreshold = confidenceThreshold;
        NmsThreshold = nmsThreshold;
    }

    /// <inheritdoc/>
    public float DetectionThreshold { get; set; }

    /// <inheritdoc/>
    public float ConfidenceThreshold { get; set; }

    /// <inheritdoc/>
    public float NmsThreshold { get; set; }

    /// <summary>
    /// Gets labels.
    /// </summary>
    public static readonly string[] Labels = ["Face"];


    /// <inheritdoc/>
    public SKRectI[] Forward(SKBitmap image)
    {
        return ToBoxes(ForwardDetection(image));
    }

    /// <inheritdoc/>
    public FaceDetectionResult[] ForwardDetection(SKBitmap image)
    {
        var rgb = BitmapToRgbFloat(image);
        return ForwardDetection(rgb);
    }

    /// <inheritdoc/>
    public SKRectI[] Forward(SKBitmap image, SKRectI rectangle, bool clamp = true)
    {
        return ForwardDetection(image, rectangle, clamp).Select(static x => x.Box).ToArray();
    }

    /// <inheritdoc/>
    public FaceDetectionResult[] ForwardDetection(SKBitmap image, SKRectI rectangle, bool clamp = true)
    {
        if (image is null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        var roi = rectangle;
        if (clamp)
        {
            var l = Math.Max(0, roi.Left);
            var t = Math.Max(0, roi.Top);
            var r = Math.Min(image.Width, roi.Right);
            var b = Math.Min(image.Height, roi.Bottom);
            if (r <= l || b <= t)
            {
                return [];
            }

            roi = new SKRectI(l, t, r, b);
        }
        else if (roi.Left < 0 || roi.Top < 0 || roi.Right > image.Width || roi.Bottom > image.Height)
        {
            throw new ArgumentOutOfRangeException(nameof(rectangle), "Rectangle is outside image bounds.");
        }

        using var subset = new SKBitmap();
        if (!image.ExtractSubset(subset, roi))
        {
            return [];
        }

        var detections = ForwardDetection(subset);
        var offset = new SKPointI(roi.Left, roi.Top);
        var translated = new FaceDetectionResult[detections.Length];
        for (int i = 0; i < detections.Length; i++)
        {
            var detection = detections[i];
            translated[i] = new FaceDetectionResult
            {
                Id = detection.Id,
                Score = detection.Score,
                Rectangle = detection.Rectangle.Add(offset),
                Points = TranslateLandmarks(detection.Points, offset)
            };
        }

        return translated;
    }

    /// <inheritdoc/>
    public SKRectI[] Forward(float[][,] image)
    {
        return ToBoxes(ForwardDetection(image));
    }

    /// <inheritdoc/>
    public FaceDetectionResult[] ForwardDetection(float[][,] image)
    {
        if (image.Length != 3)
        {
            throw new ArgumentException("Image must have 3 channels (RGB)");
        }

        var width  = image[0].GetLength(1);
        var height = image[0].GetLength(0);

        var targetWidth = _inputWidth;
        var targetHeight = _inputHeight;

        float[] inputData;
        if (_yoloOutputMode)
        {
            var resized = new float[3][,];
            for (int i = 0; i < image.Length; i++)
            {
                resized[i] = ResizePreservedChannel(image[i], targetHeight, targetWidth, 0.0f, out _, out _, out _);
            }

            inputData = MergeChannelsScaled(resized, scale: 1.0f / 255.0f, offset: 0.0f);
        }
        else
        {
            var resized = new float[3][,];
            for (int i = 0; i < image.Length; i++)
            {
                resized[i] = ResizeChannel(image[i], targetHeight, targetWidth);
            }

            // pre-processing + merge in one pass: (pixel - 127) / 128, CHW layout
            inputData = MergeChannelsNormalized(resized);
        }

        // session run
        var dimensions = new int[] { 1, 3, targetHeight, targetWidth };
        var t = new DenseTensor<float>(inputData, dimensions);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, t) };

        using var outputs = _session.Run(inputs);
        var results = outputs.ToArray();

        // This repository currently supports both model layouts:
        // 1) split outputs (confidences + boxes), and
        // 2) YOLO-style single output (bbox + landmarks + classes).
        var candidates = results.Length switch
        {
            1 => ParseYoloOutput(results[0].AsTensor<float>(), width, height, targetWidth, targetHeight),
            _ => ParseSplitOutput(results, width, height)
        };

        return ApplyNms(candidates).ToArray();
    }

    private List<FaceDetectionResult> ParseSplitOutput(DisposableNamedOnnxValue[] outputs, int sourceWidth, int sourceHeight)
    {
        var confidences = outputs[0].AsTensor<float>().ToArray();
        var boxes = outputs[1].AsTensor<float>().ToArray();
        var length = confidences.Length;

        var candidates = new List<FaceDetectionResult>(length / 2);
        for (int i = 0, j = 0; i < length; i += 2, j += 4)
        {
            var score = confidences[i + 1];
            if (score > DetectionThreshold)
            {
                candidates.Add(new FaceDetectionResult
                {
                    Rectangle = new SKRectI(
                        (int)(boxes[j + 0] * sourceWidth),
                        (int)(boxes[j + 1] * sourceHeight),
                        (int)(boxes[j + 2] * sourceWidth),
                        (int)(boxes[j + 3] * sourceHeight)
                    ),
                    Id = 0,
                    Score = score
                });
            }
        }

        return candidates;
    }

    private List<FaceDetectionResult> ParseYoloOutput(Tensor<float> output, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
    {
        var vector = output.ToArray();
        var dimensions = output.Dimensions;
        var count = dimensions.Length >= 3 ? dimensions[^1] : (Labels.Length + 15);
        var length = dimensions.Length >= 3 ? dimensions[^2] : (vector.Length / count);
        var transposed = false;
        if (dimensions.Length >= 3 && count > 32 && length > 0 && length <= 32)
        {
            transposed = true;
            (count, length) = (length, count);
        }

        var classes = Labels.Length;
        var yoloSquare = count - classes;

        if (length <= 0)
        {
            return [];
        }

        var gain = Math.Min((float)targetWidth / sourceWidth, (float)targetHeight / sourceHeight);
        var padX = (targetWidth - (sourceWidth * gain)) / 2.0f;
        var padY = (targetHeight - (sourceHeight * gain)) / 2.0f;

        var candidates = new List<FaceDetectionResult>(length);
        for (int i = 0; i < length; i++)
        {
            float Read(int index)
            {
                return transposed
                    ? vector[(index * length) + i]
                    : vector[(i * count) + index];
            }

            if (count == 6)
            {
                var score = Read(4);
                if (score <= DetectionThreshold)
                {
                    continue;
                }

                var yolo26Rect = new SKRectI(
                    (int)((Read(0) - padX) / gain),
                    (int)((Read(1) - padY) / gain),
                    (int)((Read(2) - padX) / gain),
                    (int)((Read(3) - padY) / gain)
                );

                candidates.Add(new FaceDetectionResult
                {
                    Rectangle = Clamp(yolo26Rect, sourceWidth, sourceHeight),
                    Id = (int)MathF.Round(Read(5)),
                    Score = score
                });

                continue;
            }

            if (yoloSquare < 15)
            {
                continue;
            }

            var objectness = Read(4);
            if (objectness <= DetectionThreshold)
            {
                continue;
            }

            var bestClass = 0;
            var bestScore = float.MinValue;
            for (int c = 0; c < classes; c++)
            {
                var classScore = Read(yoloSquare + c);
                if (classScore > bestScore)
                {
                    bestScore = classScore;
                    bestClass = c;
                }
            }

            if (bestScore <= ConfidenceThreshold)
            {
                continue;
            }

            var cx = Read(0);
            var cy = Read(1);
            var w = Read(2);
            var h = Read(3);

            var rect = new SKRectI(
                (int)((cx - (w / 2.0f) - padX) / gain),
                (int)((cy - (h / 2.0f) - padY) / gain),
                (int)((cx + (w / 2.0f) - padX) / gain),
                (int)((cy + (h / 2.0f) - padY) / gain)
            );

            var landmarks = new SKPointI[5];
            for (int p = 0; p < 5; p++)
            {
                var px = Read(5 + (2 * p));
                var py = Read(6 + (2 * p));
                landmarks[p] = new SKPointI(
                    (int)((px - padX) / gain),
                    (int)((py - padY) / gain)
                );
            }

            candidates.Add(new FaceDetectionResult
            {
                Rectangle = Clamp(rect, sourceWidth, sourceHeight),
                Id = bestClass,
                Score = bestScore,
                Points = new Face5Landmarks(landmarks)
            });
        }

        return candidates;
    }

    private List<FaceDetectionResult> ApplyNms(List<FaceDetectionResult> candidates)
    {
        candidates.Sort(static (a, b) => b.Score.CompareTo(a.Score));
        var suppressed = new bool[candidates.Count];
        var picked = new List<FaceDetectionResult>(candidates.Count);

        for (int i = 0; i < candidates.Count; i++)
        {
            if (suppressed[i] || candidates[i].Score <= ConfidenceThreshold)
            {
                continue;
            }

            var first = candidates[i];
            picked.Add(first);

            for (int j = i + 1; j < candidates.Count; j++)
            {
                if (!suppressed[j] && first.Box.IoU(candidates[j].Box) > NmsThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        return picked;
    }

    private static (string InputName, int InputWidth, int InputHeight) ReadInputInfo(InferenceSession session)
    {
        var kv = session.InputMetadata.First();
        var dims = kv.Value.Dimensions;

        // ponytail: face detector expects NCHW. Fallback keeps current behavior if metadata is missing.
        var h = dims.Length > 2 && dims[2] > 0 ? dims[2] : 480;
        var w = dims.Length > 3 && dims[3] > 0 ? dims[3] : 640;

        return (kv.Key, w, h);
    }

    private static bool IsYoloOutputModel(InferenceSession session)
    {
        return session.OutputMetadata.Count == 1;
    }

    private static string GetModelFileName(FaceDetectorModel model)
    {
        return model switch
        {
            FaceDetectorModel.FaceOnnx => "face_detector_640.onnx",
            FaceDetectorModel.Yolov5 => "yolov5s-face.onnx",
            FaceDetectorModel.Yolo26 => "yolo26_face_fp16.onnx",
            _ => throw new ArgumentOutOfRangeException(nameof(model), model, "Unsupported face detector model.")
        };
    }

    private static string ResolveModelPath(string modelFileName)
    {
        var relative = Path.Combine("Models", modelFileName);
        if (File.Exists(relative))
        {
            return relative;
        }

        var baseDirPath = Path.Combine(AppContext.BaseDirectory, "Models", modelFileName);
        if (File.Exists(baseDirPath))
        {
            return baseDirPath;
        }

        return relative;
    }

    /// <summary>
    /// Converts an SKBitmap to per-channel float arrays [R, G, B][height, width] in 0–255 range.
    /// </summary>
    private static float[][,] BitmapToRgbFloat(SKBitmap bitmap)
    {
        int w = bitmap.Width, h = bitmap.Height;
        var r = new float[h, w];
        var g = new float[h, w];
        var b = new float[h, w];

        var pixels = bitmap.Pixels; // SKColor[] — always RGBA regardless of internal format
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                var px = pixels[(y * w) + x];
                r[y, x] = px.Red;
                g[y, x] = px.Green;
                b[y, x] = px.Blue;
            }
        }

        return [r, g, b];
    }

    /// <summary>
    /// Bilinear resize of a single-channel float[height, width] array.
    /// </summary>
    private static float[,] ResizeChannel(float[,] src, int newH, int newW)
    {
        int srcH = src.GetLength(0), srcW = src.GetLength(1);
        var dst = new float[newH, newW];
        float scaleY = (float)srcH / newH;
        float scaleX = (float)srcW / newW;

        for (int y = 0; y < newH; y++)
        {
            float sy = y * scaleY;
            int y0 = (int)sy;
            int y1 = Math.Min(y0 + 1, srcH - 1);
            float fy = sy - y0;

            for (int x = 0; x < newW; x++)
            {
                float sx = x * scaleX;
                int x0 = (int)sx;
                int x1 = Math.Min(x0 + 1, srcW - 1);
                float fx = sx - x0;

                dst[y, x] =
                    (src[y0, x0] * (1 - fy) * (1 - fx)) +
                    (src[y0, x1] * (1 - fy) * fx) +
                    (src[y1, x0] * fy        * (1 - fx)) +
                    (src[y1, x1] * fy        * fx);
            }
        }

        return dst;
    }

    private static float[,] ResizePreservedChannel(float[,] src, int newH, int newW, float fill, out float gain, out float padX, out float padY)
    {
        int srcH = src.GetLength(0), srcW = src.GetLength(1);
        gain = Math.Min((float)newW / srcW, (float)newH / srcH);
        var resizedH = Math.Max(1, (int)Math.Round(srcH * gain));
        var resizedW = Math.Max(1, (int)Math.Round(srcW * gain));
        padX = (newW - resizedW) / 2.0f;
        padY = (newH - resizedH) / 2.0f;

        var dst = new float[newH, newW];
        for (int y = 0; y < newH; y++)
        {
            for (int x = 0; x < newW; x++)
            {
                dst[y, x] = fill;
            }
        }

        var resized = ResizeChannel(src, resizedH, resizedW);
        var x0 = (int)padX;
        var y0 = (int)padY;
        for (int y = 0; y < resizedH; y++)
        {
            for (int x = 0; x < resizedW; x++)
            {
                dst[y + y0, x + x0] = resized[y, x];
            }
        }

        return dst;
    }

    /// <summary>
    /// Merges per-channel arrays into a normalized flat CHW float array suitable for the ONNX tensor.
    /// </summary>
    private static float[] MergeChannelsNormalized(float[][,] channels)
    {
        int c = channels.Length;
        int h = channels[0].GetLength(0);
        int w = channels[0].GetLength(1);
        var result = new float[c * h * w];
        int idx = 0;
        for (int ci = 0; ci < c; ci++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    result[idx++] = (channels[ci][y, x] - 127.0f) / 128.0f;
                }
            }
        }

        return result;
    }

    private static float[] MergeChannelsScaled(float[][,] channels, float scale, float offset)
    {
        int c = channels.Length;
        int h = channels[0].GetLength(0);
        int w = channels[0].GetLength(1);
        var result = new float[c * h * w];
        int idx = 0;
        for (int ci = 0; ci < c; ci++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    result[idx++] = (channels[ci][y, x] * scale) + offset;
                }
            }
        }

        return result;
    }

    private static SKRectI Clamp(SKRectI rectangle, int width, int height)
    {
        var l = Math.Clamp(rectangle.Left, 0, width);
        var t = Math.Clamp(rectangle.Top, 0, height);
        var r = Math.Clamp(rectangle.Right, 0, width);
        var b = Math.Clamp(rectangle.Bottom, 0, height);
        return r <= l || b <= t ? SKRectI.Empty : new SKRectI(l, t, r, b);
    }

    private static SKRectI[] ToBoxes(FaceDetectionResult[] detections)
    {
        var boxes = new SKRectI[detections.Length];
        for (int i = 0; i < detections.Length; i++)
        {
            boxes[i] = detections[i].Box;
        }

        return boxes;
    }

    private static Face5Landmarks? TranslateLandmarks(Face5Landmarks? points, SKPointI offset)
    {
        if (points is null)
        {
            return null;
        }

        var source = points.All;
        var translated = new SKPointI[source.Length];
        for (int i = 0; i < source.Length; i++)
        {
            translated[i] = new SKPointI(source[i].X + offset.X, source[i].Y + offset.Y);
        }

        return new Face5Landmarks(translated);
    }

    private bool _disposed;

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _session?.Dispose();
            }

            _disposed = true;
        }
    }

    ~FaceDetector()
    {
        Dispose(false);
    }

}
