using SkiaSharp;

namespace FaceONNX;

/// <summary>
/// Defines face detector interface.
/// </summary>
public interface IFaceDetector : IDisposable
{
    #region Interface

    /// <summary>
    /// Gets or sets confidence threshold.
    /// </summary>
    float ConfidenceThreshold { get; set; }

    /// <summary>
    /// Gets or sets NonMaxSuppression threshold.
    /// </summary>
    float NmsThreshold { get; set; }

    /// <summary>
    /// Returns face detection results.
    /// </summary>
    /// <param name="image">Bitmap</param>
    /// <returns>Rectangles</returns>
    SKRectI[] Forward(SKBitmap image);

    /// <summary>
    /// Returns face detection results.
    /// </summary>
    /// <param name="image">Image in RGB terms as float channel arrays [channel][height, width]</param>
    /// <returns>Rectangles</returns>
    SKRectI[] Forward(float[][,] image);

    #endregion
}
