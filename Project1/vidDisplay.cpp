/*
  Astitva Goel
  January 2026
  CS 5330 Computer Vision - Project 1

  Features:
  - Multiple image filters (grayscale, blur, edge detection, etc.)
  - Face detection with Haar cascades
  - Creative effects (embossing, selective blur/color)
  - Video recording capability
  - Meme caption generator

  Controls:
    q - Quit
    s - Save current frame
    r - Toggle video recording
    c - Add captions (meme mode)
    d - Remove captions
    g - Grayscale (OpenCV cvtColor)
    h - Grayscale (custom max channel)
    t - Sepia tone
    n - Blur (naive 2D at method - slow)
    b - Blur (separable ptr method - fast)
    x - Sobel X edge detection
    y - Sobel Y edge detection
    m - Gradient magnitude
    i - Color quantization
    f - Face detection
    1 - Focus on face (blur background)
    2 - Embossing effect
    3 - Grayscale background with color face
    e - Erosion effect
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

/*
  Gets current system time in seconds with microsecond precision.
  Used for performance timing of filter operations.

  Returns: current time as double (seconds.microseconds)
*/
double getTime() {
    struct timeval cur;
    gettimeofday(&cur, NULL);
    return cur.tv_sec + cur.tv_usec / 1000000.0;
}

int main(){
    // Open default camera (device 0)
    VideoCapture cap(0);
    if (cap.isOpened() == false){
        cout << "Cannot open the video camera" << endl;
        cin.get();
        return -1;
    }

    // Get camera properties for recording and display
    double fps = cap.get(CAP_PROP_FPS);
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    
    cout << "Frames per second: " << fps << endl;
    cout << "Resolution: " << frame_width << "x" << frame_height << endl;

    // Mat objects for storing frames and intermediate filter results
    Mat camVideo;           // Raw camera frame
    Mat grayVideo;          // Grayscale converted frame
    Mat gaussianVideo;      // Blurred frame
    Mat sobelxVideo;        // Sobel X gradient (signed)
    Mat sobelyVideo;        // Sobel Y gradient (signed)
    Mat magnitudeVideo;     // Combined gradient magnitude
    Mat sobelxVis;          // Sobel X visualization (unsigned)
    Mat sobelyVis;          // Sobel Y visualization (unsigned)
    Mat magnitudeVis;       // Magnitude visualization (unsigned)
    Mat quantizedVideo;     // Color-quantized frame
    Mat erodeVideo;         // Eroded frame
    Mat embossVid;          // Embossed frame
    Mat sepiaVideo;         // Sepia-toned frame

    // Filter mode flags - only one can be active at a time
    bool grayMode = false;          // OpenCV grayscale
    bool altGrayMode = false;       // Custom max-channel grayscale
    bool sepiaMode = false;         // Vintage sepia tone
    bool gaussianMode = false;      // Optimized separable blur
    bool gaussianMode1 = false;     // Naive 2D blur (for timing comparison)
    bool sobelXMode = false;        // Horizontal gradient (vertical edges)
    bool sobelYMode = false;        // Vertical gradient (horizontal edges)
    bool magnitudeMode = false;     // Combined edge magnitude
    bool quantizeMode = false;      // Posterization effect
    bool faceDetectMode = false;    // Face detection with boxes
    bool focusMode = false;         // Blur background, sharp face
    bool erodeMode = false;         // Morphological erosion
    bool embossMode = false;        // 3D relief effect
    bool grayfocusMode = false;     // Gray background, color face

    // Video recording state
    bool isRecording = false;
    VideoWriter videoWriter;

    // Meme caption state
    string topCaption = "";
    string bottomCaption = "";
    bool captionsEnabled = false;

    // Print available controls
    cout << "\n=== VIDEO CONTROLS ===" << endl;
    cout << "q - Quit | s - Save frame | r - Toggle recording" << endl;
    cout << "g/h - Grayscale | t - Sepia" << endl;
    cout << "n - Blur (2D at method - SLOW) | b - Blur (Separable ptr - FAST)" << endl;
    cout << "x/y - Sobel | m - Magnitude | i - Quantize" << endl;
    cout << "f - Face | 1 - Focus | 2 - Emboss | 3 - Color face" << endl;
    cout << "e - Erode | c - Captions | d - Remove captions" << endl;

    // Main processing loop - runs continuously for video
    while (true) {
        // Capture new frame from camera
        cap.read(camVideo);
        if(camVideo.empty()){
            cout << "Video camera is disconnected" << endl;
            cin.get();
            break;
        }

        // Process frame based on active filter mode
        // displayFrame will hold the final output
        Mat displayFrame;

        if(grayMode){
            // OpenCV's standard weighted grayscale conversion
            cvtColor(camVideo, grayVideo, COLOR_BGR2GRAY);
            cvtColor(grayVideo, displayFrame, COLOR_GRAY2BGR);  // Convert back to 3-channel
        }
        else if(altGrayMode){
            // Custom grayscale using maximum channel value
            grayscale(camVideo, displayFrame);
        }
        else if(sepiaMode){
            // Vintage photograph sepia tone effect
            sepiatone(camVideo, displayFrame);
        }
        else if(gaussianMode1){
            // Naive 2D Gaussian blur with timing (SLOW - uses at<> method)
            double start = getTime();
            blur5x5_1(camVideo, displayFrame);
            double end = getTime();
            cout << "blur5x5_1 (at method): " << (end - start) << " sec" << endl;
        }
        else if(gaussianMode){
            // Optimized separable Gaussian blur with timing (FAST - uses ptr<> method)
            double start = getTime();
            blur5x5_2(camVideo, displayFrame);
            double end = getTime();
            cout << "blur5x5_2 (separable ptr): " << (end - start) << " sec" << endl;
        }
        else if(sobelXMode){
            // Horizontal gradients - detects vertical edges
            sobelX3x3(camVideo, sobelxVideo);
            convertScaleAbs(sobelxVideo, displayFrame);  // Convert signed to unsigned for display
        }
        else if(sobelYMode){
            // Vertical gradients - detects horizontal edges
            sobelY3x3(camVideo, sobelyVideo);
            convertScaleAbs(sobelyVideo, displayFrame);  // Convert signed to unsigned for display
        }
        else if(magnitudeMode){
            // Combine X and Y gradients for omnidirectional edge detection
            sobelX3x3(camVideo, sobelxVideo);
            sobelY3x3(camVideo, sobelyVideo);
            magnitude(sobelxVideo, sobelyVideo, magnitudeVideo);
            displayFrame = magnitudeVideo.clone();
        }
        else if(quantizeMode){
            // Posterization: reduce color levels for cartoon effect
            blur5x5_2(camVideo, gaussianVideo);  // Blur first to reduce noise
            quantize(gaussianVideo, displayFrame, 10);  // Quantize to 10 levels
        }
        else if(faceDetectMode){
            // Detect faces and draw bounding boxes
            std::vector<Rect> faces;
            cvtColor(camVideo, grayVideo, COLOR_BGR2GRAY);  // Face detection needs grayscale
            detectFaces(grayVideo, faces);
            displayFrame = camVideo.clone();
            drawBoxes(displayFrame, faces);
        }
        else if(focusMode){
            // Portrait mode: blur background, keep face sharp
            std::vector<Rect> faces;
            cvtColor(camVideo, grayVideo, COLOR_BGR2GRAY);
            detectFaces(grayVideo, faces);
            blurBackgroundExceptFace(camVideo, displayFrame, faces);
            drawBoxes(displayFrame, faces);
        }
        else if(erodeMode){
            // Morphological erosion - shrinks bright regions
            erodeVid(camVideo, displayFrame);
        }
        else if(embossMode){
            // 3D relief effect using directional lighting simulation
            sobelX3x3(camVideo, sobelxVideo);
            sobelY3x3(camVideo, sobelyVideo);
            embossingEffect(sobelxVideo, sobelyVideo, displayFrame);
        }
        else if(grayfocusMode){
            // Selective color: grayscale background, color face
            std::vector<Rect> faces;
            cvtColor(camVideo, grayVideo, COLOR_BGR2GRAY);
            detectFaces(grayVideo, faces);
            grayBackgroundExceptFace(camVideo, displayFrame, faces);
            drawBoxes(displayFrame, faces);
        }
        else{
            // No filter active - display original camera feed
            displayFrame = camVideo.clone();
        }
        
        // Overlay meme captions if enabled
        if(captionsEnabled) {
            addCaptionToFrame(displayFrame, topCaption, bottomCaption);
        }
        
        // Handle video recording
        if(isRecording && videoWriter.isOpened()) {
            // Write current frame to video file
            videoWriter.write(displayFrame);
            
            // Show recording indicator on screen (not saved to file)
            Mat displayWithIndicator = displayFrame.clone();
            circle(displayWithIndicator, Point(30, 30), 10, Scalar(0, 0, 255), -1);  // Red dot
            putText(displayWithIndicator, "REC", Point(50, 40), 
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            imshow("Camera Video Stream", displayWithIndicator);
        }
        else {
            // Normal display without recording indicator
            imshow("Camera Video Stream", displayFrame);
        }

        // Check for keyboard input (1ms timeout for real-time video)
        int k = waitKey(1);

        // === KEYBOARD HANDLERS ===
        
        if (k == 'q') {
            // Quit: cleanup recording if active
            if(isRecording && videoWriter.isOpened()) {
                videoWriter.release();
                cout << "Recording saved to output.avi" << endl;
            }
            break;
        }
        
        if (k == 's') {
            // Save current frame to JPEG file
            imwrite("sample_copy.jpg", displayFrame);
            cout << "Image saved as sample_copy.jpg" << endl;
        }
        
        if(k == 'r') {
            // Toggle video recording on/off
            if(!isRecording) {
                // Start recording
                int codec = VideoWriter::fourcc('M','J','P','G');  // Motion JPEG codec
                videoWriter.open("output.avi", codec, fps, 
                                Size(frame_width, frame_height), true);
                
                if(videoWriter.isOpened()) {
                    isRecording = true;
                    cout << "Recording started..." << endl;
                } else {
                    cout << "Failed to start recording!" << endl;
                }
            }
            else {
                // Stop recording and save file
                isRecording = false;
                videoWriter.release();
                cout << "Recording stopped and saved to output.avi" << endl;
            }
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
        
        // === FILTER TOGGLE KEYS ===
        // Each filter turns off all others to ensure only one is active
        
        if (k == 'g'){
            // Toggle OpenCV grayscale conversion
            grayMode = !grayMode;
            altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Gray mode: " << (grayMode ? "ON":"OFF") << endl;
        }
        
        if (k == 'h'){
            // Toggle custom grayscale (max channel method)
            altGrayMode = !altGrayMode;
            grayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Alternate Gray mode: " << (altGrayMode ? "ON":"OFF") << endl;
        }
        
        if (k == 't'){
            // Toggle sepia tone vintage effect
            sepiaMode = !sepiaMode;
            grayMode = altGrayMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Sepia tone: " << (sepiaMode ? "ON":"OFF") << endl;
        }
        
        if (k == 'n'){
            // Toggle naive 2D blur with performance timing
            gaussianMode1 = !gaussianMode1;
            grayMode = altGrayMode = sepiaMode = gaussianMode = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Gaussian Blur (naive at method): " << (gaussianMode1 ? "ON":"OFF") << endl;
        }
        
        if (k == 'b'){
            // Toggle optimized separable blur with performance timing
            gaussianMode = !gaussianMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Gaussian Blur (separable ptr method): " << (gaussianMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'x'){
            // Toggle Sobel X (detects vertical edges)
            sobelXMode = !sobelXMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Sobel X: " << (sobelXMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'y'){
            // Toggle Sobel Y (detects horizontal edges)
            sobelYMode = !sobelYMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Sobel Y: " << (sobelYMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'm'){
            // Toggle gradient magnitude (all edges)
            magnitudeMode = !magnitudeMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Magnitude: " << (magnitudeMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'i'){
            // Toggle color quantization (posterization)
            quantizeMode = !quantizeMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = false;
            faceDetectMode = focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Quantization: " << (quantizeMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'f'){
            // Toggle face detection with bounding boxes
            faceDetectMode = !faceDetectMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            focusMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Face Detection: " << (faceDetectMode ? "ON":"OFF") << endl;
        }
        
        if(k == '1'){
            // Toggle portrait mode (blur background, sharp face)
            focusMode = !focusMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = erodeMode = embossMode = grayfocusMode = false;
            cout << "Focus on Face: " << (focusMode ? "ON":"OFF") << endl;
        }
        
        if(k == '2'){
            // Toggle embossing 3D relief effect
            embossMode = !embossMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = erodeMode = grayfocusMode = false;
            cout << "Embossing Effect: " << (embossMode ? "ON":"OFF") << endl;
        }
        
        if(k == '3'){
            // Toggle selective color (gray background, color face)
            grayfocusMode = !grayfocusMode;
            embossMode = grayMode = altGrayMode = sepiaMode = false;
            gaussianMode = gaussianMode1 = sobelXMode = sobelYMode = false;
            magnitudeMode = quantizeMode = faceDetectMode = focusMode = erodeMode = false;
            cout << "Gray Background, Color Face: " << (grayfocusMode ? "ON":"OFF") << endl;
        }
        
        if(k == 'e'){
            // Toggle morphological erosion
            erodeMode = !erodeMode;
            grayMode = altGrayMode = sepiaMode = gaussianMode = gaussianMode1 = false;
            sobelXMode = sobelYMode = magnitudeMode = quantizeMode = false;
            faceDetectMode = focusMode = embossMode = grayfocusMode = false;
            cout << "Erode: " << (erodeMode ? "ON":"OFF") << endl;
        }
    }

    // Cleanup: release camera and close windows
    cap.release();
    destroyAllWindows();
    return 0;
}