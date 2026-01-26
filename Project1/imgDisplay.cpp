/*
  Astitva Goel
  January 2026
  CS 5330 Computer Vision - Project 1

  Features:
  - Multiple image filters (grayscale, blur, edge detection, etc.)
  - Face detection with Haar cascades
  - Creative effects (embossing, selective blur/color)
  - Meme caption generator
  - Performance timing for blur algorithms

  Controls:
    q - Quit
    s - Save current image
    0 - Reset to original
    g - Grayscale (OpenCV cvtColor)
    h - Grayscale (custom max channel)
    t - Sepia tone
    n - Blur (naive 2D at method - slow, with timing)
    b - Blur (separable ptr method - fast, with timing)
    x - Sobel X edge detection
    y - Sobel Y edge detection
    m - Gradient magnitude
    i - Color quantization
    f - Face detection
    1 - Focus on face (blur background)
    2 - Embossing effect
    3 - Grayscale background with color face
    e - Erosion effect
    c - Add captions (meme mode)
    d - Remove captions
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "filters.h"
#include "faceDetect.h"

using namespace cv;
using namespace std;


double getTime() {
    struct timeval cur;
    gettimeofday(&cur, NULL);
    return cur.tv_sec + cur.tv_usec / 1000000.0;
}

int main() {
    // Load image from file
    string imagePath = "/Users/astitva/Documents/PRCV/Project1/sample.jpg";
    Mat originalImage = imread(imagePath);
    
    // Verify image loaded successfully
    if(originalImage.empty()) {
        cout << "Image not found at: " << imagePath << endl;
        return 1;
    }
    
    // Display image properties
    cout << "Image size: " << originalImage.rows << "x" << originalImage.cols << endl;
    cout << "Channels: " << (int)originalImage.channels() << endl;
    cout << "Depth: " << (int)originalImage.elemSize()/originalImage.channels() << " bytes" << endl;
    
    // Filter mode flags - only one can be active at a time
    bool grayMode = false;          // OpenCV weighted grayscale
    bool altGrayMode = false;       // Custom max-channel grayscale
    bool sepiaMode = false;         // Vintage sepia tone
    bool gaussianMode = false;      // Optimized separable blur (fast)
    bool gaussianMode1 = false;     // Naive 2D blur (slow) - for timing comparison
    bool sobelXMode = false;        // Horizontal gradient (vertical edges)
    bool sobelYMode = false;        // Vertical gradient (horizontal edges)
    bool magnitudeMode = false;     // Combined edge magnitude
    bool quantizeMode = false;      // Posterization effect
    bool faceDetectMode = false;    // Face detection with boxes
    bool focusMode = false;         // Blur background, sharp face
    bool erodeMode = false;         // Morphological erosion
    bool embossMode = false;        // 3D relief effect
    bool grayfocusMode = false;     // Gray background, color face
    
    // Caption state for meme generator
    string topCaption = "";
    string bottomCaption = "";
    bool captionsEnabled = false;
    
    // Print available controls to terminal
    cout << "\n=== CONTROLS ===" << endl;
    cout << "q - Quit | s - Save | 0 - Reset" << endl;
    cout << "g/h - Grayscale | t - Sepia" << endl;
    cout << "n - Blur (2D at method - SLOW) | b - Blur (Separable ptr method - FAST)" << endl;
    cout << "x/y - Sobel | m - Magnitude | i - Quantize" << endl;
    cout << "f - Face detect | 1 - Focus | 2 - Emboss" << endl;
    cout << "3 - Color face | e - Erode | c - Captions | d - Remove captions" << endl;
    
    // Main loop - waits indefinitely for keypress between iterations
    while (true) {
        // Apply selected filter to original image
        // Always process from originalImage to avoid degradation
        Mat processedImage;
        
        if(grayMode) {
            // OpenCV's standard weighted grayscale (0.299*R + 0.587*G + 0.114*B)
            Mat gray;
            cvtColor(originalImage, gray, COLOR_BGR2GRAY);
            cvtColor(gray, processedImage, COLOR_GRAY2BGR);  // Convert back to 3-channel
        }
        else if(altGrayMode) {
            // Custom grayscale using maximum channel value
            grayscale(originalImage, processedImage);
        }
        else if(sepiaMode) {
            // Vintage sepia tone effect
            sepiatone(originalImage, processedImage);
        }
        else if(gaussianMode1) {
            // Naive 2D Gaussian blur with timing (SLOW - uses at<> method)
            double start = getTime();
            blur5x5_1(originalImage, processedImage);
            double end = getTime();
            cout << "blur5x5_1 (at method) took: " << (end - start) << " seconds" << endl;
        }
        else if(gaussianMode) {
            // Optimized separable Gaussian blur with timing (FAST - uses ptr<> method)
            double start = getTime();
            blur5x5_2(originalImage, processedImage);
            double end = getTime();
            cout << "blur5x5_2 (separable ptr) took: " << (end - start) << " seconds" << endl;
        }
        else if(sobelXMode) {
            // Horizontal gradients - detects vertical edges
            Mat sobelX;
            sobelX3x3(originalImage, sobelX);
            convertScaleAbs(sobelX, processedImage);  // Convert signed to unsigned for display
        }
        else if(sobelYMode) {
            // Vertical gradients - detects horizontal edges
            Mat sobelY;
            sobelY3x3(originalImage, sobelY);
            convertScaleAbs(sobelY, processedImage);  // Convert signed to unsigned for display
        }
        else if(magnitudeMode) {
            // Combine X and Y gradients using Euclidean distance
            Mat sobelX, sobelY, mag;
            sobelX3x3(originalImage, sobelX);
            sobelY3x3(originalImage, sobelY);
            magnitude(sobelX, sobelY, mag);
            processedImage = mag.clone();
        }
        else if(quantizeMode) {
            // Posterization: blur then reduce color levels
            Mat blurred;
            blur5x5_2(originalImage, blurred);
            quantize(blurred, processedImage, 10);
        }
        else if(faceDetectMode) {
            // Detect faces and draw bounding boxes
            Mat gray;
            std::vector<Rect> faces;
            cvtColor(originalImage, gray, COLOR_BGR2GRAY);  // Face detection needs grayscale
            detectFaces(gray, faces);
            processedImage = originalImage.clone();
            drawBoxes(processedImage, faces);
        }
        else if(focusMode) {
            // Portrait mode: blur background, keep face sharp
            Mat gray;
            std::vector<Rect> faces;
            cvtColor(originalImage, gray, COLOR_BGR2GRAY);
            detectFaces(gray, faces);
            blurBackgroundExceptFace(originalImage, processedImage, faces);
            drawBoxes(processedImage, faces);
        }
        else if(erodeMode) {
            // Morphological erosion - shrinks bright regions
            erodeVid(originalImage, processedImage);
        }
        else if(embossMode) {
            // 3D relief effect using directional lighting
            Mat sobelX, sobelY;
            sobelX3x3(originalImage, sobelX);
            sobelY3x3(originalImage, sobelY);
            embossingEffect(sobelX, sobelY, processedImage);
        }
        else if(grayfocusMode) {
            // Selective color: grayscale background, color face
            Mat gray;
            std::vector<Rect> faces;
            cvtColor(originalImage, gray, COLOR_BGR2GRAY);
            detectFaces(gray, faces);
            grayBackgroundExceptFace(originalImage, processedImage, faces);
            drawBoxes(processedImage, faces);
        }
        else {
            // No filter active - display original image
            processedImage = originalImage.clone();
        }
        
        // Overlay meme captions if enabled
        if(captionsEnabled) {
            addCaptionToFrame(processedImage, topCaption, bottomCaption);
        }
        
        // Display image and wait for keypress
        imshow("Image Display", processedImage);
        int k = waitKey(0);  // Wait indefinitely for key (static image mode)
        
        // === KEYBOARD HANDLERS ===
        
        if (k == 'q') {
            // Quit program
            break;
        }
        
        if (k == 's') {
            // Save current processed image to file
            imwrite("sample_copy.jpg", processedImage);
            cout << "Image saved as sample_copy.jpg" << endl;
        }
        
        if (k == '0') {
            // Reset all filters and captions to original state
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            captionsEnabled = false;
            topCaption = bottomCaption = "";
            cout << "Reset to original" << endl;
        }
        
        // === FILTER TOGGLE KEYS ===
        // Each filter turns off all others to ensure only one is active
        
        if (k == 'g') {
            // Toggle OpenCV weighted grayscale
            grayMode = !grayMode;
            altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Gray mode: " << (grayMode ? "ON":"OFF") << endl;
        }
        
        if (k == 'h') {
            // Toggle custom max-channel grayscale
            altGrayMode = !altGrayMode;
            grayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Alternate Gray mode: " << (altGrayMode ? "ON":"OFF") << endl;
        }
        
        if (k == 't') {
            // Toggle sepia tone vintage effect
            sepiaMode = !sepiaMode;
            grayMode = altGrayMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Sepia tone: " << (sepiaMode ? "ON":"OFF") << endl;
        }
        
        if (k == 'n') {
            // Toggle naive 2D blur with performance timing
            gaussianMode1 = !gaussianMode1;
            grayMode = altGrayMode = sepiaMode = gaussianMode = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Gaussian Blur (naive at method): " << (gaussianMode1 ? "ON":"OFF") << endl;
        }
        
        if (k == 'b') {
            // Toggle optimized separable blur with performance timing
            gaussianMode = !gaussianMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Gaussian Blur (separable ptr method): " << (gaussianMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'x') {
            // Toggle Sobel X (detects vertical edges)
            sobelXMode = !sobelXMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Sobel X: " << (sobelXMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'y') {
            // Toggle Sobel Y (detects horizontal edges)
            sobelYMode = !sobelYMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Sobel Y: " << (sobelYMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'm') {
            // Toggle gradient magnitude (all edges)
            magnitudeMode = !magnitudeMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Magnitude: " << (magnitudeMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'i') {
            // Toggle color quantization (posterization)
            quantizeMode = !quantizeMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Quantization: " << (quantizeMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'f') {
            // Toggle face detection with bounding boxes
            faceDetectMode = !faceDetectMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Face Detection: " << (faceDetectMode ? "ON":"OFF") << endl;
        }
        
        if(k == '1') {
            // Toggle portrait mode (blur background, sharp face)
            focusMode = !focusMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Focus on Face: " << (focusMode ? "ON":"OFF") << endl;
        }
        
        if(k == '2') {
            // Toggle embossing 3D relief effect
            embossMode = !embossMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = grayfocusMode = false;
            cout << "Embossing Effect: " << (embossMode ? "ON":"OFF") << endl;
        }
        
        if(k == '3') {
            // Toggle selective color (gray background, color face)
            grayfocusMode = !grayfocusMode;
            embossMode = grayMode = altGrayMode = sepiaMode = false;
            gaussianMode = gaussianMode1 = sobelXMode = sobelYMode = false;
            magnitudeMode = quantizeMode = faceDetectMode = focusMode = erodeMode = false;
            cout << "Gray Background, Color Face: " << (grayfocusMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'e') {
            // Toggle morphological erosion
            erodeMode = !erodeMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = embossMode = grayfocusMode = false;
            cout << "Erode: " << (erodeMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'c') {
            // Meme caption input mode
            cout << "\n=== MEME GENERATOR ===" << endl;
            topCaption = getCaptionFromUser("Enter top caption (or press Enter to skip): ");
            bottomCaption = getCaptionFromUser("Enter bottom caption (or press Enter to skip): ");
            captionsEnabled = true;
            cout << "Captions added! Press 'c' again to edit or 'd' to remove." << endl;
        }
        
        if(k == 'd') {
            // Remove captions
            captionsEnabled = false;
            topCaption = "";
            bottomCaption = "";
            cout << "Captions removed" << endl;
        }
    }
    
    // Cleanup: close all windows before exit
    destroyAllWindows();
    return 0;
}