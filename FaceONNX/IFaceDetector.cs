using SkiaSharp;

namespace FaceONNX;

/// <summary>
/// Defines face detector interface.
/// </summary>
public interface IFaceDetector : IDisposable
{
    #region Interface

    /// <summary>
    /// The minimum confidence score required to consider a detected region as a face.
    /// </summary>
    float DetectionThreshold { get; set; }

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
    /// Returns rich face detection results.
    /// </summary>
    /// <param name="image">Bitmap</param>
    /// <returns>Face detections</returns>
    FaceDetectionResult[] ForwardDetection(SKBitmap image);

    /// <summary>
    /// Returns face detection results for a region of interest.
    /// </summary>
    /// <param name="image">Bitmap</param>
    /// <param name="rectangle">Region of interest</param>
    /// <param name="clamp">Clamp ROI to image bounds when true</param>
    /// <returns>Rectangles</returns>
    SKRectI[] Forward(SKBitmap image, SKRectI rectangle, bool clamp = true);

    /// <summary>
    /// Returns rich face detection results for a region of interest.
    /// </summary>
    /// <param name="image">Bitmap</param>
    /// <param name="rectangle">Region of interest</param>
    /// <param name="clamp">Clamp ROI to image bounds when true</param>
    /// <returns>Face detections</returns>
    FaceDetectionResult[] ForwardDetection(SKBitmap image, SKRectI rectangle, bool clamp = true);

    /// <summary>
    /// Returns face detection results.
    /// </summary>
    /// <param name="image">Image in RGB terms as float channel arrays [channel][height, width]</param>
    /// <returns>Rectangles</returns>
    SKRectI[] Forward(float[][,] image);

    /// <summary>
    /// Returns rich face detection results.
    /// </summary>
    /// <param name="image">Image in RGB terms as float channel arrays [channel][height, width]</param>
    /// <returns>Face detections</returns>
    FaceDetectionResult[] ForwardDetection(float[][,] image);

    #endregion
}
