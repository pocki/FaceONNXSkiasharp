# Face Detection and Multiple Face Recognition with FaceOnnx

This project showcases the implementation of face detection and multi-face recognition capabilities using the FaceOnnx library. With FaceOnnx, you can accurately detect faces within images or video streams and perform advanced recognition of multiple faces simultaneously. This powerful solution is ideal for applications requiring facial recognition, from security systems to user authentication, leveraging the robust capabilities of the FaceOnnx library to enhance accuracy and efficiency.

Changed Dependencies:

- Removed: Emgu.CV, OpenCvSharp4, UMapx, System.Drawing
- Added: SkiaSharp 4
- Updated to .NET 10.0

Using only SkiaSharp 4 for image processing and rendering, this project demonstrates how to integrate face detection and recognition functionalities seamlessly. SkiaSharp provides a versatile graphics library that allows for efficient image manipulation, making it an excellent choice for handling the visual aspects of face recognition tasks.

These changes streamline the project by reducing dependencies and focusing on a single, powerful graphics library, ensuring better performance, better cross-platform compatibility and maintainability. The transition to .NET 10.0 also brings improved performance and access to the latest features of the .NET ecosystem.