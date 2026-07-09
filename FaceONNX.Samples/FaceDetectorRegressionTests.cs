using FaceONNX;
using SkiaSharp;

namespace FaceONNX.Samples;

internal static class FaceDetectorRegressionTests
{
    public static int Run()
    {
        Console.WriteLine("FaceONNX.Samples: running regression tests");

        using var faceDetector = new FaceDetector(0.95f, 0.5f);
        var imagePaths = Directory.GetFiles("images", "*.jpg");
        if (imagePaths.Length == 0)
        {
            Console.WriteLine("FAIL: no source images found in images/");
            return 1;
        }

        var failed = false;
        foreach (var imagePath in imagePaths)
        {
            var fileName = Path.GetFileName(imagePath);
            var expectedPath = Path.Combine("results", fileName);
            if (!File.Exists(expectedPath))
            {
                Console.WriteLine($"FAIL: expected baseline missing: {expectedPath}");
                failed = true;
                continue;
            }

            using var source = SKBitmap.Decode(imagePath);
            using var expected = SKBitmap.Decode(expectedPath);
            if (source is null || expected is null)
            {
                Console.WriteLine($"FAIL: unable to decode {fileName} or baseline");
                failed = true;
                continue;
            }

            var faces = faceDetector.Forward(source);
            using var actual = RenderDetections(source, faces);
            var (meanAbsDiff, maxAbsDiff) = ComputeRgbDiff(actual, expected);

            // JPEG baselines can vary slightly by platform build, keep a tiny tolerance.
            var pass = meanAbsDiff <= 2.0f;
            Console.WriteLine(
                $"{(pass ? "PASS" : "FAIL")}: {fileName} faces={faces.Length} meanDiff={meanAbsDiff:F3} maxDiff={maxAbsDiff}");
            if (!pass)
            {
                failed = true;
            }
        }

        return failed ? 1 : 0;
    }

    private static SKBitmap RenderDetections(SKBitmap source, SKRectI[] faces)
    {
        using var surface = SKSurface.Create(new SKImageInfo(source.Width, source.Height));
        var canvas = surface.Canvas;
        canvas.DrawBitmap(source, 0, 0, SKSamplingOptions.Default);

        using var boxPaint = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            Color = SKColors.Yellow,
            StrokeWidth = 4,
            IsAntialias = true,
        };

        foreach (var rect in faces)
        {
            canvas.DrawRect(new SKRect(rect.Left, rect.Top, rect.Right, rect.Bottom), boxPaint);
        }

        using var image = surface.Snapshot();
        using var data = image.Encode(SKEncodedImageFormat.Jpeg, 95);
        return SKBitmap.Decode(data)!;
    }

    private static (float MeanAbsDiff, byte MaxAbsDiff) ComputeRgbDiff(SKBitmap actual, SKBitmap expected)
    {
        if (actual.Width != expected.Width || actual.Height != expected.Height)
        {
            return (float.MaxValue, byte.MaxValue);
        }

        var total = 0L;
        byte max = 0;
        var pixels = actual.Width * actual.Height;
        var actualPixels = actual.Pixels;
        var expectedPixels = expected.Pixels;
        for (var i = 0; i < pixels; i++)
        {
            var a = actualPixels[i];
            var e = expectedPixels[i];
            var dr = (byte)Math.Abs(a.Red - e.Red);
            var dg = (byte)Math.Abs(a.Green - e.Green);
            var db = (byte)Math.Abs(a.Blue - e.Blue);

            total += dr + dg + db;
            if (dr > max) max = dr;
            if (dg > max) max = dg;
            if (db > max) max = db;
        }

        var mean = (float)total / (pixels * 3);
        return (mean, max);
    }
}
