/*
  Astitva Goel
  January 2026
  CS 5330 Computer Vision

  Implementation of various image filtering and effect functions
  including edge detection, blurring, morphological operations,
  and creative effects.
*/


#include "filters.h"
#include <iostream>
#include <sys/time.h> 
using namespace cv;
using namespace std;
double getTime() {
    struct timeval cur;
    gettimeofday(&cur, NULL);
    return cur.tv_sec + cur.tv_usec / 1000000.0;
}
/*
  Grayscale function
  Converts a color image to grayscale using maximum channel value.
  For each pixel, finds the maximum of R, G, B and assigns to all channels.
*/
int grayscale(Mat &src, Mat &dst);
int grayscale(Mat &src, Mat &dst){
    dst = Mat(src.size(), CV_8UC3 );
    // Process each pixel
    for(int i=0; i<src.rows; i++){
        Vec3b *ptr = src.ptr<Vec3b>(i);
        for(int j=0; j<src.cols;j++){
            // uchar red = ptr[j][2];
            // Taking maximum of the three color channels
            uchar gray = std::max(ptr[j][0], std::max(ptr[j][1], ptr[j][2]));
            // dst.at<Vec3b>(i,j) = Vec3b(gray, gray, gray);
            // Assigning the same value to all three channels
            dst.ptr<Vec3b>(i)[j] = Vec3b(gray, gray, gray);
        }
    }
    return 0;
}

/*
  SepiaTone function
  Applies sepia tone filter for vintage photograph effect.
  Uses weighted combination of RGB channels per standard sepia formula.
*/
int sepiatone(Mat &src, Mat &dst);
int sepiatone(Mat &src, Mat &dst){
    dst = Mat(src.size(), CV_8UC3);
    for(int i=0; i<src.rows; i++){
        Vec3b *ptr = src.ptr<Vec3b>(i);
        for(int j=0; j<src.cols; j++){
            // Get original BGR values
            float blue = ptr[j][0];
            float green = ptr[j][1];
            float red = ptr[j][2];
            // Apply sepia transformation
            float newBlue = 0.272*red + 0.534*green + 0.131*blue;
            float newGreen =0.349*red + 0.686*green + 0.168*blue;
            float newRed = 0.393*red + 0.769*green + 0.189*blue;
            // Clamp values to valid range [0, 255]
            if(newBlue > 255) newBlue = 255;
            if(newGreen > 255) newGreen = 255;
            if(newRed > 255) newRed = 255;

            dst.ptr<Vec3b>(i)[j] = Vec3b(newBlue, newGreen, newRed);
        }
    }
    return 0;
}
/*
  Gaussian Blur Function
  Applies 5x5 Gaussian blur using full 2D convolution.
  Uses approximation of Gaussian kernel with integer weights.
  This is slower using at() method but demonstrates the full 2D convolution approach.
*/
int blur5x5_1(Mat &src, Mat &dst);
int blur5x5_1(Mat &src, Mat &dst){
    double start = getTime();
    // copy the source image to destination image for allocation
    src.copyTo(dst);
    for(int i=2; i<src.rows-2; i++){
        for(int j=2; j<src.cols-2; j++){
            for(int c=0; c<3; c++){
                 // Computing weighted sum of 5x5 neighborhood
                int value = src.at<Vec3b>(i-2,j-2)[c] + 2*src.at<Vec3b>(i-2,j-1)[c] + 4*src.at<Vec3b>(i-2,j)[c] + 2*src.at<Vec3b>(i-2,j+1)[c] + src.at<Vec3b>(i-2,j+2)[c]
                          + 2*src.at<Vec3b>(i-1,j-2)[c] + 4*src.at<Vec3b>(i-1,j-1)[c] + 8*src.at<Vec3b>(i-1,j)[c] + 4*src.at<Vec3b>(i-1,j+1)[c] + 2*src.at<Vec3b>(i-1,j+2)[c]
                          + 4*src.at<Vec3b>(i,j-2)[c] + 8*src.at<Vec3b>(i,j-1)[c] + 16*src.at<Vec3b>(i,j)[c] + 8*src.at<Vec3b>(i,j+1)[c] + 4*src.at<Vec3b>(i,j+2)[c]
                          + 2*src.at<Vec3b>(i+1,j-2)[c] + 4*src.at<Vec3b>(i+1,j-1)[c] + 8*src.at<Vec3b>(i+1,j)[c] + 4*src.at<Vec3b>(i+1,j+1)[c] + 2*src.at<Vec3b>(i+1,j+2)[c]
                          + src.at<Vec3b>(i+2,j-2)[c] + 2*src.at<Vec3b>(i+2,j-1)[c] + 4*src.at<Vec3b>(i+2,j)[c] + 2*src.at<Vec3b>(i+2,j+1)[c] + src.at<Vec3b>(i+2,j+2)[c];
                // Normalize by dividing by sum of weights (100)
                dst.at<Vec3b>(i,j)[c] = value/100;
            }
        }
    }
    double end = getTime();
    cout << "blur5x5_1 (at method): " << (end - start) << " sec" << endl;
    return 0;
}
/*
  Seperable Gaussian Blur function
  Applies 5x5 Gaussian blur using separable filters (optimized version).
  Takes advantage of the fact that 2D Gaussian can be separated into
  two 1D convolutions (horizontal then vertical), reducing computation using pointers.
*/
int blur5x5_2(Mat &src, Mat &dst);
int blur5x5_2(Mat &src, Mat &dst){
    double start = getTime();
    Mat temp;
    src.copyTo(temp);
    // First pass: horizontal blur with 1D kernel [1 2 4 2 1]
    for(int i=0; i<src.rows;i++){
        Vec3b *sptr_mid = src.ptr<Vec3b>(i);
        Vec3b *tptr = temp.ptr<Vec3b>(i);
        for(int j=2; j<src.cols-2;j++){
            for(int c=0; c<3; c++){
                int value = sptr_mid[j-2][c] +
                            2*sptr_mid[j-1][c] +
                            4*sptr_mid[j][c]+
                            2*sptr_mid[j+1][c]+
                            sptr_mid[j+2][c];
                tptr[j][c] = value/10;
            }
        }
    }
    temp.copyTo(dst);
    // Second pass: vertical blur with 1D kernel [1 2 4 2 1]
    for(int i=2; i<src.rows-2; i++){
        Vec3b *dptr = dst.ptr<Vec3b>(i);
        Vec3b *sptr_up1 = temp.ptr<Vec3b>(i-2);
        Vec3b *sptr_up = temp.ptr<Vec3b>(i-1);
        Vec3b *sptr_mid = temp.ptr<Vec3b>(i);
        Vec3b *sptr_down = temp.ptr<Vec3b>(i+1);
        Vec3b *sptr_down1 = temp.ptr<Vec3b>(i+2);
        for(int j=0; j<src.cols; j++){
            for(int c=0; c<3; c++){
                int value = sptr_up1[j][c]+
                            2*sptr_up[j][c]+
                            4*sptr_mid[j][c]+
                            2*sptr_down[j][c]+
                            sptr_down1[j][c];
                dptr[j][c] = value/10;
            }
        }
    }
    double end = getTime();
    cout << "blur5x5_2 (separable ptr): " << (end - start) << " sec" << endl;
    return 0;

}

/*
  SobelX function
  Computes horizontal gradients using 3x3 Sobel X operator.
  Uses separable filtering: vertical smoothing [1 2 1], then horizontal derivative [-1 0 1].
  Output is signed 16-bit to preserve negative gradient values.
*/
int sobelX3x3(Mat &src, Mat &dst);
int sobelX3x3(Mat &src, Mat &dst){
    dst.create(src.size(), CV_16SC3);
    Mat temp(src.size(), CV_16SC3);
    // First pass: vertical smoothing with [1 2 1] kernel
    for(int i=1; i<src.rows-1; i++){
        // need the upper, mid and down pointers for vertical matrix
        Vec3b *sptr_up = src.ptr<Vec3b>(i-1);
        Vec3b *sptr_mid = src.ptr<Vec3b>(i);
        Vec3b *sptr_down = src.ptr<Vec3b>(i+1);
        Vec3s *tptr = temp.ptr<Vec3s>(i);
        for(int j=0; j<src.cols; j++){
            for(int c=0; c<3; c++){
                int value = sptr_up[j][c] + 2*sptr_mid[j][c] + sptr_down[j][c];
                tptr[j][c] = value;
            }
        }
    }
    // Second pass: horizontal derivative with [-1 0 1] kernel
    for(int i=1; i<src.rows-1; i++){
        Vec3s *tptr = temp.ptr<Vec3s>(i);
        Vec3s *dptr = dst.ptr<Vec3s>(i);
        for(int j=1; j<src.cols-1; j++){
            for(int c=0; c<3; c++){
                int value = -tptr[j-1][c] + tptr[j+1][c];
                dptr[j][c] = value;
            }
        }
    }
    return 0;
}
/*
  SobelY function
  Computes vertical gradients using 3x3 Sobel Y operator.
  Uses separable filtering: horizontal smoothing [1 2 1], then vertical derivative [1 0 -1].
  Output is signed 16-bit to preserve negative gradient values.
*/
int sobelY3x3(Mat &src, Mat &dst);
int sobelY3x3(Mat &src, Mat &dst){
    dst.create(src.size(), CV_16SC3);
    Mat temp(src.size(), CV_16SC3);
    // First pass: horizontal smoothing with [1 2 1] kernel
     for (int i = 0; i < src.rows; i++) {
        Vec3b *sptr = src.ptr<Vec3b>(i);
        Vec3s *tptr = temp.ptr<Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                tptr[j][c] =
                    sptr[j-1][c] +
                    2 * sptr[j][c] +
                    sptr[j+1][c];
            }
        }
    }

    // Vertical derivative pass using [ 1 0 -1 ]
    for (int i = 1; i < src.rows - 1; i++) {
        Vec3s *up   = temp.ptr<Vec3s>(i-1);
        Vec3s *down = temp.ptr<Vec3s>(i+1);
        Vec3s *dptr = dst.ptr<Vec3s>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                dptr[j][c] = up[j][c] - down[j][c];
            }
        }
    }
    return 0;
}
/*
  Magnitude Function
  Computes gradient magnitude using Euclidean distance formula.
  Combines X and Y gradients: magnitude = sqrt(sobelX^2 + sobelY^2)
  Results are converted to unsigned 8-bit for display.
*/
int magnitude(Mat &sobelX, Mat &sobelY, Mat &dst);
int magnitude(Mat &sobelX, Mat &sobelY, Mat &dst){
    dst.create(sobelX.size(), CV_8UC3);
    for(int i=0; i<sobelX.rows; i++){
        Vec3s *sxptr = sobelX.ptr<Vec3s>(i);
        Vec3s *syptr = sobelY.ptr<Vec3s>(i);
        Vec3b *dptr = dst.ptr<Vec3b>(i);
        for(int j=0; j<sobelX.cols; j++){
            for(int c=0; c<3; c++){
                // Computing Euclidean magnitude: sqrt(sobelX^2 + sobelY^2)
                int value = sqrt(sxptr[j][c] * sxptr[j][c] + syptr[j][c] * syptr[j][c]);
                if(value>255) value = 255;
                dptr[j][c] = (uchar)value;
            }
        }
    }
    return 0;
}
/*
  Quantization function
  Reduces color palette by quantizing intensity levels.
  Divides 0-255 range into specified number of discrete levels.
*/
int quantize(Mat &src, Mat &dst, int levels);
int quantize(Mat &src, Mat &dst, int levels){
    dst = Mat(src.size(), CV_8UC3);
    // Calculating bucket size for quantization
    int b = 255/levels;
    for(int i=0; i<src.rows; i++){
        Vec3b *sptr = src.ptr<Vec3b>(i);
        Vec3b *dptr = dst.ptr<Vec3b>(i);
        for(int j=0; j<src.cols; j++){
            for(int c=0; c<3; c++){
                 // Quantize: divide by bucket size, multiply back
                int value = (sptr[j][c]/b)*b;
                dptr[j][c] = value;
            }
        }
    }
    return 0;
}
/*
  Portrait Mode function
  Creates portrait mode effect by blurring background and keeping faces sharp.
*/
int blurBackgroundExceptFace(Mat &src, Mat &dst, std::vector<Rect> &faces);
int blurBackgroundExceptFace(Mat &src, Mat &dst, std::vector<Rect> &faces) {
    // Blur the entire image
    Mat blurred;
    blur5x5_2(src, blurred);
    // Copy blurred to destination
    blurred.copyTo(dst);
    // Copy original (sharp) face regions back on it
    for(int i = 0; i < faces.size(); i++) {
        if(faces[i].width > 50) {  // minimum face size
            // Extract the face region from original image
            Rect faceRect = faces[i];
            faceRect.x = max(0, faceRect.x);
            faceRect.y = max(0, faceRect.y);
            faceRect.width = min(faceRect.width, src.cols - faceRect.x);
            faceRect.height = min(faceRect.height, src.rows - faceRect.y);
            // Copy sharp face region onto blurred background
            src(faceRect).copyTo(dst(faceRect));
        }
    }
    return 0;
}
/*
  Erosion function
  Applies morphological erosion using 5x5 window.
  For each pixel, replaces with minimum value in neighborhood.
  This shrinks bright regions and expands dark regions.
*/
int erodeVid(Mat &src, Mat &dst);
int erodeVid(Mat &src, Mat &dst){
    src.copyTo(dst);
    for(int i=2; i<src.rows-2; i++){
        for(int j=2; j<src.cols-2; j++){
            for(int c=0; c<3; c++){
                int value = std::min(src.at<Vec3b>(i-2,j-2)[c],
                                std::min(src.at<Vec3b>(i-2,j-1)[c],
                                std::min(src.at<Vec3b>(i-2,j)[c],
                                std::min(src.at<Vec3b>(i-2,j+1)[c],
                                src.at<Vec3b>(i-2,j+2)[c]))));
                dst.at<Vec3b>(i,j)[c] = value;
            }
        }
    }
    return 0;
}

/*
  Embossing Function
  Creates embossing effect by computing directional derivative.
  Uses dot product of gradient with light direction vector (0.7071, 0.7071)
  to simulate lighting from 45-degree angle, creating 3D relief appearance.
*/
int embossingEffect(Mat &sobelX, Mat &sobelY, Mat &dst);
int embossingEffect(Mat &sobelX, Mat &sobelY, Mat &dst){
    dst.create(sobelX.size(), CV_8UC3);
    for(int i=0; i<sobelX.rows; i++){
        Vec3s *sxptr = sobelX.ptr<Vec3s>(i);
        Vec3s *syptr = sobelY.ptr<Vec3s>(i);
        Vec3b *dptr = dst.ptr<Vec3b>(i);
        for(int j=0; j<sobelX.cols; j++){
            for(int c=0; c<3; c++){
                int value = sxptr[j][c] * 0.7071 + syptr[j][c] * 0.7071;
                if(value>255) value = 255;
                dptr[j][c] = (uchar)value;
            }
        }
    }
    return 0;
}
/*
  Gray Background function
  Converts background to grayscale while preserving face color.
  Similar to portrait function
*/
int grayBackgroundExceptFace(Mat &src, Mat &dst, std::vector<Rect> &faces);
int grayBackgroundExceptFace(Mat &src, Mat &dst, std::vector<Rect> &faces) {
    // Blur the entire image
    Mat grayed;
    grayscale(src, grayed);
    //  Copy blurred to destination
    grayed.copyTo(dst);
    // Copy original (sharp) face regions back
    for(int i = 0; i < faces.size(); i++) {
        if(faces[i].width > 50) {  // minimum face size
            // Extract the face region from original image
            Rect faceRect = faces[i];
            faceRect.x = max(0, faceRect.x);
            faceRect.y = max(0, faceRect.y);
            faceRect.width = min(faceRect.width, src.cols - faceRect.x);
            faceRect.height = min(faceRect.height, src.rows - faceRect.y);
            // Copy sharp face region onto blurred background
            src(faceRect).copyTo(dst(faceRect));
        }
    }
    return 0;
}

/*
  Caption function
  Prompts user for text input from terminal.
*/
std::string getCaptionFromUser(const std::string& prompt) {
    std::cout << prompt;
    std::string caption;
    getline(std::cin, caption);
    return caption;
}

/*
  Caption function
  Overlays meme-style captions on image with classic white text and black outline.
  Text is centered horizontally at top and bottom of frame.
*/
void addCaptionToFrame(Mat &frame, const std::string& topText, const std::string& bottomText) {
    if(topText.empty() && bottomText.empty()) return;
    // Meme-style settings
    int fontFace = cv::FONT_HERSHEY_DUPLEX;
    double fontScale = 1.5;
    int thickness = 3;
    int baseline = 0;

    // White text with black outline
    Scalar textColor(255, 255, 255);
    Scalar outlineColor(0, 0, 0);

    // Add top text using topPosition function of Point class in OpenCV for pixel manipulation
    if(!topText.empty()) {
        Size textSize = getTextSize(topText, fontFace, fontScale, thickness, &baseline);
        Point topPosition((frame.cols - textSize.width) / 2, textSize.height + 30);

        // Use putText function to write text
        putText(frame, topText, topPosition, fontFace, fontScale,
                outlineColor, thickness + 2, LINE_AA);
        putText(frame, topText, topPosition, fontFace, fontScale,
                textColor, thickness, LINE_AA);
    }

    // Add bottom text
    if(!bottomText.empty()) {
        Size textSize = getTextSize(bottomText, fontFace, fontScale, thickness, &baseline);
        Point bottomPosition((frame.cols - textSize.width) / 2,
                            frame.rows - 30);

        putText(frame, bottomText, bottomPosition, fontFace, fontScale,
                outlineColor, thickness + 2, LINE_AA);

        putText(frame, bottomText, bottomPosition, fontFace, fontScale,
                textColor, thickness, LINE_AA);
    }
}