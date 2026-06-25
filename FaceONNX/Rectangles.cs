using SkiaSharp;

namespace FaceONNX;

/// <summary>
/// Using for face boxes operations.
/// </summary>
public static class Rectangles
{
    /// <summary>
    /// Returns processed rectangle.
    /// </summary>
    /// <param name="rectangle">Rectangle</param>
    /// <param name="point">Point</param>
    /// <returns>Rectangle</returns>
    public static SKRectI Add(this SKRectI rectangle, SKPointI point)
    {
        return SKRectI.Create(rectangle.Left + point.X, rectangle.Top + point.Y, rectangle.Width, rectangle.Height);
    }

    /// <summary>
    /// Returns processed rectangle.
    /// </summary>
    /// <param name="rectangle">Rectangle</param>
    /// <param name="point">Point</param>
    /// <returns>Rectangle</returns>
    public static SKRectI Sub(this SKRectI rectangle, SKPointI point)
    {
        return SKRectI.Create(rectangle.Left - point.X, rectangle.Top - point.Y, rectangle.Width, rectangle.Height);
    }

    /// <summary>
    /// Returns processed rectangles.
    /// </summary>
    /// <param name="rectangles">Rectangles</param>
    /// <param name="point">Point</param>
    /// <returns>Rectangles</returns>
    public static SKRectI[] Add(this SKRectI[] rectangles, SKPointI point)
    {
        var count = rectangles.Length;
        var output = new SKRectI[count];

        for (int i = 0; i < count; i++)
        {
            output[i] = rectangles[i].Add(point);
        }

        return output;
    }

    /// <summary>
    /// Returns processed rectangles.
    /// </summary>
    /// <param name="rectangles">Rectangles</param>
    /// <param name="point">Point</param>
    /// <returns>Rectangles</returns>
    public static SKRectI[] Sub(this SKRectI[] rectangles, SKPointI point)
    {
        var count = rectangles.Length;
        var output = new SKRectI[count];

        for (int i = 0; i < count; i++)
        {
            output[i] = rectangles[i].Sub(point);
        }

        return output;
    }

    /// <summary>
    /// Returns four points from rectangle.
    /// </summary>
    /// <param name="rectangle">Rectangle</param>
    /// <returns>Points</returns>
    public static SKPointI[] ToPoints(this SKRectI rectangle)
    {
        return
        [
            new(rectangle.Left,  rectangle.Top),
            new(rectangle.Right, rectangle.Top),
            new(rectangle.Right, rectangle.Bottom),
            new(rectangle.Left,  rectangle.Bottom)
        ];
    }

    /// <summary>
    /// Returns rectangle from four points.
    /// </summary>
    /// <param name="points"></param>
    /// <returns></returns>
    public static SKRectI FromPoints(this SKPointI[] points)
    {
        if (points.Length != 4)
        {
            throw new ArgumentException("A rectangle can only be built using four points.");
        }

        return new SKRectI(points[0].X, points[0].Y, points[2].X, points[2].Y);
    }

    /// <summary>
    /// Returns point from rectangle.
    /// </summary>
    /// <param name="rectangle">Rectangle</param>
    /// <returns>Point</returns>
    public static SKPointI GetPoint(this SKRectI rectangle)
    {
        return new SKPointI(rectangle.Left, rectangle.Top);
    }

    /// <summary>
    /// Returns size area.
    /// </summary>
    /// <param name="size">Size</param>
    /// <returns>Area</returns>
    public static int Area(this SKSizeI size)
    {
        return size.Width * size.Height;
    }

    /// <summary>
    /// Returns rectangle area.
    /// </summary>
    /// <param name="rectangle">Rectangle</param>
    /// <returns>Area</returns>
    public static int Area(this SKRectI rectangle)
    {
        return rectangle.Width * rectangle.Height;
    }

    /// <summary>
    /// Returns the maximum rectangle.
    /// </summary>
    /// <param name="rectangles">Rectangles</param>
    /// <returns>Rectangle</returns>
    public static SKRectI Max(params SKRectI[] rectangles)
    {
        // params
        var length = rectangles.Length;
        var rectangle = SKRectI.Empty;
        var area = 0;
        var max = 0;

        // do job
        for (int i = 0; i < length; i++)
        {
            rectangle = rectangles[i];

            if (rectangle.IsEmpty)
            {
                continue;
            }

            if (rectangle.Area() > area)
            {
                max = i;
            }
        }

        // output
        return length > 0 ? rectangles[max] : rectangle;
    }

    /// <summary>
    /// Returns the minimum rectangle.
    /// </summary>
    /// <param name="rectangles">Rectangles</param>
    /// <returns>Rectangle</returns>
    public static SKRectI Min(params SKRectI[] rectangles)
    {
        // params
        var length = rectangles.Length;
        var rectangle = SKRectI.Empty;
        var area = int.MaxValue;
        var min = int.MaxValue;

        // do job
        for (int i = 0; i < length; i++)
        {
            rectangle = rectangles[i];

            if (rectangle.IsEmpty)
            {
                continue;
            }

            if (rectangle.Area() < area)
            {
                min = i;
            }
        }

        // output
        return length > 0 ? rectangles[min] : rectangle;
    }

    /// <summary>
    /// Returns rectangle scaled to box.
    /// </summary>
    /// <param name="rectangle">Rectangle</param>
    /// <returns>Rectangle</returns>
    public static SKRectI ToBox(this SKRectI rectangle)
    {
        var maxDim = Math.Max(rectangle.Width, rectangle.Height);
        var dx = maxDim - rectangle.Width;
        var dy = maxDim - rectangle.Height;

        return SKRectI.Create(
            rectangle.Left - (dx / 2),
            rectangle.Top  - (dy / 2),
            rectangle.Width  + dx,
            rectangle.Height + dy);
    }

    /// <summary>
    /// Returns rectangle scaled to box.
    /// </summary>
    /// <param name="rectangle">Rectangle</param>
    /// <param name="scale">Factor</param>
    /// <returns>Rectangle</returns>
    public static SKRectI ToBox(this SKRectI rectangle, float scale)
    {
        float gainX = rectangle.Width  * scale;
        float gainY = rectangle.Height * scale;

        return SKRectI.Create(
            (int)(rectangle.Left - (gainX / 2)),
            (int)(rectangle.Top  - (gainY / 2)),
            (int)(rectangle.Width  + gainX),
            (int)(rectangle.Height + gainY));
    }

    /// <summary>
    /// Returns rectangle scaled to box.
    /// </summary>
    /// <param name="rectangles">Rectangle</param>
    /// <returns>Rectangle</returns>
    public static SKRectI[] ToBox(params SKRectI[] rectangles)
    {
        int length = rectangles.Length;
        var newRectangles = new SKRectI[length];

        for (int i = 0; i < length; i++)
        {
            newRectangles[i] = rectangles[i].ToBox();
        }

        return newRectangles;
    }

    /// <summary>
    /// Returns rectangle scaled to box with image size.
    /// </summary>
    /// <param name="factor">Factor</param>
    /// <param name="rectangles">Rectangles</param>
    /// <returns>Rectangle</returns>
    public static SKRectI[] ToBox(float factor, params SKRectI[] rectangles)
    {
        int length = rectangles.Length;
        var newRectangles = new SKRectI[length];

        for (int i = 0; i < length; i++)
        {
            newRectangles[i] = rectangles[i].ToBox(factor);
        }

        return newRectangles;
    }

    /// <summary>
    /// Implements IoU operator.
    /// </summary>
    /// <param name="a">First rectangle</param>
    /// <param name="b">Second rectangle</param>
    /// <returns>Value</returns>
    public static float IoU(this SKRectI a, SKRectI b)
    {
        var xA = Math.Max(a.Left, b.Left);
        var yA = Math.Max(a.Top, b.Top);
        var xB = Math.Min(a.Right, b.Right);
        var yB = Math.Min(a.Bottom, b.Bottom);

        var interArea = Math.Abs(Math.Max(xB - xA, 0) * Math.Max(yB - yA, 0));

        if (interArea == 0)
        {
            return 0;
        }

        var boxAArea = Math.Abs((a.Right - a.Left) * (float)(a.Bottom - a.Top));
        var boxBArea = Math.Abs((b.Right - b.Left) * (float)(b.Bottom - b.Top));

        return interArea / (float)(boxAArea + boxBArea - interArea);
    }

    /// <summary>
    /// Implements scale operator.
    /// </summary>
    /// <param name="rectangle">Rectangle</param>
    /// <param name="kx">Factor for x axis</param>
    /// <param name="ky">Factor for y axis</param>
    /// <returns></returns>
    public static SKRectI Scale(this SKRectI rectangle, float kx = 0.0f, float ky = 0.0f)
    {
        var x = rectangle.Left < 0 ? 0 : rectangle.Left;
        var y = rectangle.Top  < 0 ? 0 : rectangle.Top;
        var w = rectangle.Width;
        var h = rectangle.Height;

        var dw = (int)(w * kx);
        var dh = (int)(h * ky);

        return SKRectI.Create(x - (dw / 2), y - (dh / 2), w + dw, h + dh);
    }

    /// <summary>
    /// Implements scale operator.
    /// </summary>
    /// <param name="rectangle">Rectangle</param>
    /// <returns>Rectangle</returns>
    public static SKRectI Scale(this SKRectI rectangle)
    {
        var r = (int)Math.Sqrt((rectangle.Width * rectangle.Width) + (rectangle.Height * rectangle.Height));
        var dx = r - rectangle.Width;
        var dy = r - rectangle.Height;

        var x = rectangle.Left < 0 ? 0 : rectangle.Left - (dx / 2);
        var y = rectangle.Top  < 0 ? 0 : rectangle.Top  - (dy / 2);
        var w = rectangle.Width  + dx;
        var h = rectangle.Height + dy;

        return SKRectI.Create(x, y, w, h);
    }

}