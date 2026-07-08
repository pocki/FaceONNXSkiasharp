using FaceONNX;
using SkiaSharp;

namespace FaceONNX.Samples;

internal static class Program
{
    static int Main(string[] args)
    {
        if (args.Any(static x => string.Equals(x, "--test", StringComparison.OrdinalIgnoreCase)))
        {
            return FaceDetectorRegressionTests.Run();
        }

        Console.WriteLine("FaceONNX.Samples: Face detection");

        using var faceDetector = new FaceDetector(0.4f, 0.5f, FaceDetectorModel.Yolo26);

        Directory.CreateDirectory("results");

        foreach (var imagePath in Directory.GetFiles("images", "*.jpg"))
        {
            var fileName = Path.GetFileName(imagePath);

            using var bitmap = SKBitmap.Decode(imagePath);
            if (bitmap is null)
            {
                Console.WriteLine($"Failed to load: {fileName}");
                continue;
            }

            var faces = faceDetector.ForwardDetection(bitmap);

            using var surface = SKSurface.Create(new SKImageInfo(bitmap.Width, bitmap.Height));
            var canvas = surface.Canvas;
            canvas.DrawBitmap(bitmap, 0, 0, SKSamplingOptions.Default);

            using var boxPaint = new SKPaint
            {
                Style = SKPaintStyle.Stroke,
                Color = SKColors.Yellow,
                StrokeWidth = 4,
                IsAntialias = true,
            };

            foreach (var rect in faces)
            {
                canvas.DrawRect(new SKRect(rect.Rectangle.Left, rect.Rectangle.Top, rect.Rectangle.Right, rect.Rectangle.Bottom), boxPaint);
            }

            var outputPath = Path.Combine("results", fileName);
            using var image = surface.Snapshot();
            using var data = image.Encode(SKEncodedImageFormat.Jpeg, 95);
            using var stream = File.OpenWrite(outputPath);
            data.SaveTo(stream);

            Console.WriteLine($"Image: [{fileName}] --> detected [{faces.Length}] faces");
        }

        return 0;
    }
}
