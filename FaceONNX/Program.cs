using SkiaSharp;

namespace FaceONNX;

internal static class Program
{
    static void Main()
    {
        Console.WriteLine("FaceONNX: Face detection");

        using var faceDetector = new FaceDetector(0.95f, 0.5f);

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

            var faces = faceDetector.Forward(bitmap);

            using var surface = SKSurface.Create(new SKImageInfo(bitmap.Width, bitmap.Height));
            var canvas = surface.Canvas;
            canvas.DrawBitmap(bitmap, 0, 0, SKSamplingOptions.Default);

            using var boxPaint = new SKPaint
            {
                Style       = SKPaintStyle.Stroke,
                Color       = SKColors.Yellow,
                StrokeWidth = 4,
                IsAntialias = true,
            };

            foreach (var rect in faces)
            {
                canvas.DrawRect(new SKRect(rect.Left, rect.Top, rect.Right, rect.Bottom), boxPaint);
            }

            var outputPath = Path.Combine("results", fileName);
            using var image = surface.Snapshot();
            using var data  = image.Encode(SKEncodedImageFormat.Jpeg, 95);
            using var stream = File.OpenWrite(outputPath);
            data.SaveTo(stream);

            Console.WriteLine($"Image: [{fileName}] --> detected [{faces.Length}] faces");
        }
    }
}

