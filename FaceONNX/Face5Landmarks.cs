using SkiaSharp;

namespace FaceONNX;

/// <summary>
/// Defines 5-point face landmarks.
/// </summary>
public sealed class Face5Landmarks
{
    private readonly SKPointI[] _points;

    /// <summary>
    /// Initializes 5-point face landmarks.
    /// </summary>
    /// <param name="points">Five landmark points</param>
    public Face5Landmarks(SKPointI[] points)
    {
        if (points is null)
        {
            throw new ArgumentNullException(nameof(points));
        }

        if (points.Length != 5)
        {
            throw new ArgumentException("The number of face points must be 5.", nameof(points));
        }

        _points = [..points];
    }

    /// <summary>
    /// Returns all face points.
    /// </summary>
    public SKPointI[] All => [.._points];

    /// <summary>
    /// Returns right eye point.
    /// </summary>
    public SKPointI RightEye => _points[1];

    /// <summary>
    /// Returns left eye point.
    /// </summary>
    public SKPointI LeftEye => _points[0];

    /// <summary>
    /// Returns mouth points.
    /// </summary>
    public SKPointI[] Mouth => [_points[3], _points[4]];

    /// <summary>
    /// Returns nose point.
    /// </summary>
    public SKPointI Nose => _points[2];

    /// <summary>
    /// Returns face roll angle in degrees.
    /// </summary>
    public float RotationAngle
    {
        get
        {
            var dy = RightEye.Y - LeftEye.Y;
            var dx = RightEye.X - LeftEye.X;
            return (float)(Math.Atan2(dy, dx) * (180.0 / Math.PI));
        }
    }

    /// <summary>
    /// Returns simple left/right eye symmetry around nose point, in [0..1].
    /// </summary>
    public float SymmetryCoefficient
    {
        get
        {
            var dl = Distance(Nose, LeftEye);
            var dr = Distance(Nose, RightEye);
            var max = Math.Max(dl, dr);
            return max <= 0 ? 1.0f : Math.Min(dl, dr) / max;
        }
    }

    private static float Distance(SKPointI a, SKPointI b)
    {
        var dx = b.X - a.X;
        var dy = b.Y - a.Y;
        return MathF.Sqrt((dx * dx) + (dy * dy));
    }
}
