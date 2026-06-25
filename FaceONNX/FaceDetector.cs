using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace FaceONNX;

/// <summary>
/// Defines face detector.
/// </summary>
public class FaceDetector : IFaceDetector
{
    /// <summary>
    /// Inference session.
    /// </summary>
    private readonly InferenceSession _session;

    /// <summary>
    /// Initializes face detector.
    /// </summary>
    /// <param name="confidenceThreshold">Confidence threshold</param>
    /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
    public FaceDetector(float confidenceThreshold = 0.95f, float nmsThreshold = 0.5f)
    {
        _session = new InferenceSession("Models/face_detector_640.onnx");
        ConfidenceThreshold = confidenceThreshold;
        NmsThreshold = nmsThreshold;
    }

    /// <summary>
    /// Initializes face detector.
    /// </summary>
    /// <param name="options">Session options</param>
    /// <param name="confidenceThreshold">Confidence threshold</param>
    /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
    public FaceDetector(SessionOptions options, float confidenceThreshold = 0.95f, float nmsThreshold = 0.5f)
    {
        _session = new InferenceSession("Models/face_detector_640.onnx", options);
        ConfidenceThreshold = confidenceThreshold;
        NmsThreshold = nmsThreshold;
    }

    /// <inheritdoc/>
    public float ConfidenceThreshold { get; set; }

    /// <inheritdoc/>
    public float NmsThreshold { get; set; }


    /// <inheritdoc/>
    public SKRectI[] Forward(SKBitmap image)
    {
        var rgb = BitmapToRgbFloat(image);
        return Forward(rgb);
    }

    /// <inheritdoc/>
    public SKRectI[] Forward(float[][,] image)
    {
        if (image.Length != 3)
        {
            throw new ArgumentException("Image must have 3 channels (RGB)");
        }

        var width  = image[0].GetLength(1);
        var height = image[0].GetLength(0);

        const int targetWidth  = 640;
        const int targetHeight = 480;

        var resized = new float[3][,];
        for (int i = 0; i < image.Length; i++)
        {
            resized[i] = ResizeChannel(image[i], targetHeight, targetWidth);
        }

        var inputMeta = _session.InputMetadata;
        var name = inputMeta.Keys.ToArray()[0];

        // pre-processing: (pixel - 127) / 128
        SubtractMean(resized, singleArray);
        DivideBy(resized, 128.0f);
        var inputData = MergeChannels(resized);

        // session run
        var dimensions = new int[] { 1, 3, targetHeight, targetWidth };
        var t = new DenseTensor<float>(inputData, dimensions);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, t) };

        using var outputs = _session.Run(inputs);
        var results     = outputs.ToArray();
        var confidences = results[0].AsTensor<float>().ToArray();
        var boxes       = results[1].AsTensor<float>().ToArray();
        var length      = confidences.Length;

        // post-processing
        var boxes_picked = new List<SKRectI>();

        for (int i = 0, j = 0; i < length; i += 2, j += 4)
        {
            var confidence1 = confidences[i + 1];
            if (confidence1 > ConfidenceThreshold)
            {
                boxes_picked.Add(
                    new SKRectI(
                        (int)(boxes[j + 0] * width),
                        (int)(boxes[j + 1] * height),
                        (int)(boxes[j + 2] * width),
                        (int)(boxes[j + 3] * height)
                    ).ToBox());
            }
        }

        // non-max suppression
        length = boxes_picked.Count;

        for (int i = 0; i < length; i++)
        {
            var first = boxes_picked[i];

            for (int j = i + 1; j < length; j++)
            {
                var second = boxes_picked[j];
                var iou = first.IoU(second);

                if (iou > NmsThreshold)
                {
                    boxes_picked.RemoveAt(j);
                    length = boxes_picked.Count;
                    j--;
                }
            }
        }

        return boxes_picked.ToArray();
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

    /// <summary>
    /// Subtracts per-channel mean values in place.
    /// </summary>
    private static void SubtractMean(float[][,] channels, float[] mean)
    {
        for (int c = 0; c < channels.Length; c++)
        {
            var ch = channels[c];
            int h = ch.GetLength(0), w = ch.GetLength(1);
            float m = mean[c];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    ch[y, x] -= m;
                }
            }
        }
    }

    /// <summary>
    /// Divides all channel values by a scalar in place.
    /// </summary>
    private static void DivideBy(float[][,] channels, float divisor)
    {
        foreach (var ch in channels)
        {
            int h = ch.GetLength(0), w = ch.GetLength(1);
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    ch[y, x] /= divisor;
                }
            }
        }
    }

    /// <summary>
    /// Merges per-channel arrays into a flat CHW float array suitable for the ONNX tensor.
    /// </summary>
    private static float[] MergeChannels(float[][,] channels)
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
                    result[idx++] = channels[ci][y, x];
                }
            }
        }

        return result;
    }

    private bool _disposed;
    private static readonly float[] singleArray = [127.0f, 127.0f, 127.0f];

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
