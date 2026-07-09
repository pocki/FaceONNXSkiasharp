using SkiaSharp;

namespace FaceONNX;

/// <summary>
/// Extension methods for face image alignment and cropping operations.
/// </summary>
public static class FaceProcessingExtensions
{
    /// <summary>
    /// Returns aligned face.
    /// </summary>
    /// <param name="image">Bitmap</param>
    /// <param name="angle">Angle (degrees)</param>
    /// <returns>Bitmap</returns>
    public static SKBitmap Align(this SKBitmap image, float angle)
    {
        ArgumentNullException.ThrowIfNull(image);

        var output = new SKBitmap(image.Width, image.Height, image.ColorType, image.AlphaType);
        using var canvas = new SKCanvas(output);
        canvas.Clear(SKColors.Black);
        canvas.Translate(image.Width * 0.5f, image.Height * 0.5f);
        canvas.RotateDegrees(angle);
        canvas.Translate(-image.Width * 0.5f, -image.Height * 0.5f);
        canvas.DrawBitmap(image, 0, 0, SKSamplingOptions.Default);
        return output;
    }

    /// <summary>
    /// Returns aligned face.
    /// </summary>
    /// <param name="image">Bitmap</param>
    /// <param name="rectangle">Rectangle</param>
    /// <param name="angle">Angle (degrees)</param>
    /// <param name="clamp">Clamp crop or not</param>
    /// <returns>Bitmap</returns>
    public static SKBitmap Align(this SKBitmap image, SKRectI rectangle, float angle, bool clamp = true)
    {
        ArgumentNullException.ThrowIfNull(image);

        var scaledRectangle = rectangle.Scale();
        using var cropped = Crop(image, scaledRectangle, clamp);
        using var aligned = Align(cropped, angle);
        var cropRectangle = rectangle.Sub(new SKPointI(scaledRectangle.Left, scaledRectangle.Top));
        return Crop(aligned, cropRectangle, clamp);
    }

    /// <summary>
    /// Returns aligned face.
    /// </summary>
    /// <param name="image">Image in RGB terms</param>
    /// <param name="angle">Angle (degrees)</param>
    /// <returns>Image in RGB terms</returns>
    public static float[][,] Align(this float[][,] image, float angle)
    {
        var length = image.Length;

        if (length != 3)
        {
            throw new ArgumentException("Image must have 3 channels (RGB)");
        }

        var aligned = new float[length][,];
        for (int i = 0; i < length; i++)
        {
            aligned[i] = RotateChannel(image[i], -angle);
        }

        return aligned;
    }

    /// <summary>
    /// Returns aligned face.
    /// </summary>
    /// <param name="image">Image in RGB terms</param>
    /// <param name="rectangle">Rectangle</param>
    /// <param name="angle">Angle (degrees)</param>
    /// <param name="clamp">Clamp crop or not</param>
    /// <returns>Image in RGB terms</returns>
    public static float[][,] Align(this float[][,] image, SKRectI rectangle, float angle, bool clamp = true)
    {
        var length = image.Length;
        if (length != 3)
        {
            throw new ArgumentException("Image must have 3 channels (RGB)");
        }

        var scaledRectangle = rectangle.Scale();
        var cropped = new float[length][,];

        for (int i = 0; i < length; i++)
        {
            cropped[i] = CropChannel(
                image[i],
                scaledRectangle.Top,
                scaledRectangle.Left,
                scaledRectangle.Height,
                scaledRectangle.Width,
                clamp);
        }

        var aligned = Align(cropped, angle);
        var cropRectangle = rectangle.Sub(new SKPointI(scaledRectangle.Left, scaledRectangle.Top));
        var output = new float[length][,];

        for (int i = 0; i < length; i++)
        {
            output[i] = CropChannel(
                aligned[i],
                cropRectangle.Top,
                cropRectangle.Left,
                cropRectangle.Height,
                cropRectangle.Width,
                clamp);
        }

        return output;
    }

    private static SKBitmap Crop(SKBitmap image, SKRectI rectangle, bool clamp)
    {
        var roi = rectangle;
        if (clamp)
        {
            var l = Math.Max(0, roi.Left);
            var t = Math.Max(0, roi.Top);
            var r = Math.Min(image.Width, roi.Right);
            var b = Math.Min(image.Height, roi.Bottom);
            if (r <= l || b <= t)
            {
                throw new ArgumentOutOfRangeException(nameof(rectangle), "Rectangle has no overlap with image.");
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
            throw new InvalidOperationException("Failed to crop bitmap.");
        }

        return subset.Copy() ?? throw new InvalidOperationException("Failed to copy cropped bitmap.");
    }

    private static float[,] RotateChannel(float[,] src, float angleDegrees)
    {
        int h = src.GetLength(0);
        int w = src.GetLength(1);
        var dst = new float[h, w];

        float rad = angleDegrees * (MathF.PI / 180f);
        float cos = MathF.Cos(rad);
        float sin = MathF.Sin(rad);
        float cx = (w - 1) * 0.5f;
        float cy = (h - 1) * 0.5f;

        for (int y = 0; y < h; y++)
        {
            float dy = y - cy;
            for (int x = 0; x < w; x++)
            {
                float dx = x - cx;
                float sx = (cos * dx) + (sin * dy) + cx;
                float sy = (-sin * dx) + (cos * dy) + cy;
                dst[y, x] = SampleBilinear(src, sy, sx);
            }
        }

        return dst;
    }

    private static float[,] CropChannel(float[,] src, int top, int left, int height, int width, bool clamp)
    {
        int srcH = src.GetLength(0);
        int srcW = src.GetLength(1);

        if (!clamp &&
            (top < 0 || left < 0 || top + height > srcH || left + width > srcW))
        {
            throw new ArgumentOutOfRangeException(nameof(top), "Crop region is outside source bounds.");
        }

        var dst = new float[height, width];
        for (int y = 0; y < height; y++)
        {
            int sy = top + y;
            if (clamp)
            {
                sy = Math.Clamp(sy, 0, srcH - 1);
            }

            for (int x = 0; x < width; x++)
            {
                int sx = left + x;
                if (clamp)
                {
                    sx = Math.Clamp(sx, 0, srcW - 1);
                }

                dst[y, x] = src[sy, sx];
            }
        }

        return dst;
    }

    private static float SampleBilinear(float[,] src, float y, float x)
    {
        int h = src.GetLength(0);
        int w = src.GetLength(1);

        if (x < 0 || y < 0 || x > w - 1 || y > h - 1)
        {
            return 0f;
        }

        int x0 = (int)x;
        int y0 = (int)y;
        int x1 = Math.Min(x0 + 1, w - 1);
        int y1 = Math.Min(y0 + 1, h - 1);
        float dx = x - x0;
        float dy = y - y0;

        return
            (src[y0, x0] * (1 - dx) * (1 - dy)) +
            (src[y0, x1] * dx * (1 - dy)) +
            (src[y1, x0] * (1 - dx) * dy) +
            (src[y1, x1] * dx * dy);
    }
}
