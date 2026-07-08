using SkiaSharp;

namespace FaceONNX;

/// <summary>
/// Defines face detection result.
/// </summary>
public sealed class FaceDetectionResult
{
    /// <summary>
    /// Gets or sets label id.
    /// </summary>
    public int Id { get; set; }

    /// <summary>
    /// Gets or sets score.
    /// </summary>
    public float Score { get; set; }

    /// <summary>
    /// Gets or sets rectangle.
    /// </summary>
    public SKRectI Rectangle { get; set; }

    /// <summary>
    /// Gets square box.
    /// </summary>
    public SKRectI Box => Rectangle.ToBox();

    /// <summary>
    /// Gets or sets optional 5-point face landmarks.
    /// </summary>
    public Face5Landmarks? Points { get; set; }

    /// <summary>
    /// Empty detection.
    /// </summary>
    public static FaceDetectionResult Empty => new()
    {
        Rectangle = SKRectI.Empty,
        Score = 0,
        Id = -1,
        Points = new Face5Landmarks([new SKPointI(), new SKPointI(), new SKPointI(), new SKPointI(), new SKPointI()])
    };
}
