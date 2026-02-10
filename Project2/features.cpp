/*
  Astitva Goel
  Feature extraction and distance metric implementations
*/

#include "features.h"
#include <iostream>

using namespace std;
using namespace cv;

// ==================== BASELINE FEATURES ====================

// Extract 7x7 center region as feature vector
// Returns 147 values (7x7 pixels × 3 BGR channels)
vector<float> extractBaselineFeatures(const Mat &img) {
    vector<float> features;
    
    // Find image center
    int centerX = img.cols / 2;
    int centerY = img.rows / 2;
    int halfSize = 3; // 7x7 region
    
    // Extract 7x7 region around center
    for(int y = centerY - halfSize; y <= centerY + halfSize; y++) {
        for(int x = centerX - halfSize; x <= centerX + halfSize; x++) {
            Vec3b pixel = img.at<Vec3b>(y, x);
            features.push_back(pixel[0]); // Blue
            features.push_back(pixel[1]); // Green
            features.push_back(pixel[2]); // Red
        }
    }
    return features;
}

// Compute Sum of Squared Differences
// Lower distance = more similar
float computeSSD(const vector<float> &f1, const vector<float> &f2) {
    if(f1.size() != f2.size()) {
        cout << "Feature vectors must be same size!" << endl;
        return -1.0;
    }
    
    float ssd = 0.0;
    for(size_t i = 0; i < f1.size(); i++) {
        float diff = f1[i] - f2[i];
        ssd += diff * diff;
    }
    return ssd;
}

// ==================== HISTOGRAM FEATURES ====================

// Extract RG chromaticity histogram (color without brightness)
// Returns 256×256 = 65,536 values
vector<float> extractColorHistogramFeatures(const Mat &img){
    vector<float> features;
    Mat hist = Mat::zeros(256, 256, CV_32FC1);
    
    // Build 2D histogram of normalized red and green
    for(int i=0; i<img.rows; i++){
        const Vec3b *ptr = img.ptr<Vec3b>(i);
        for(int j=0; j<img.cols; j++){
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];
            
            // Avoid division by zero
            float divisor = R+G+B;
            divisor = (divisor > 0) ? divisor : 1.0;
            
            // Compute chromaticity (normalized color)
            float r = R/divisor;
            float g = G/divisor;

            // Map to histogram bins
            int rindex = (int)(r*(255)*0.5);
            int gindex = (int)(g*(255)*0.5);

            hist.at<float>(rindex,gindex) += 1.0;
        }
    }
    
    // Normalize by total pixels
    hist /= (img.cols * img.rows);
    
    // Flatten histogram to vector
    for(int i = 0; i < hist.rows; i++) {
        for(int j = 0; j < hist.cols; j++) {
            features.push_back(hist.at<float>(i, j));
        }
    }
    return features;
}

// Histogram intersection distance
// Returns 1 - overlap (0 = identical, 1 = no overlap)
float histogramIntersection(const vector<float> &h1, const vector<float> &h2) {
    if(h1.size() != h2.size()) {
        cerr << "Error: Histograms must be same size!" << endl;
        return -1.0;
    }

    float intersection = 0.0;
    
    // Sum minimum values at each bin
    for(size_t i = 0; i < h1.size(); i++) {
        intersection += min(h1[i], h2[i]);
    }
    
    // Convert to distance
    float distance = 1.0 - intersection;
    return distance;
}

// ==================== MULTI-HISTOGRAM FEATURES ====================

// Extract top and bottom half histograms
// Returns 1024 values (512 top + 512 bottom)
vector<float> extractMultiHistogramFeatures(const Mat &img) {
    vector<float> features;

    const int bins = 8;  // 8 bins per channel
    const int histSize = bins * bins * bins;  // 8×8×8 = 512

    Mat topHist = Mat::zeros(1, histSize, CV_32FC1);
    Mat bottomHist = Mat::zeros(1, histSize, CV_32FC1);

    int midRow = img.rows / 2;

    // Process top half
    for(int y = 0; y < midRow; y++) {
        const Vec3b *ptr = img.ptr<Vec3b>(y);
        for(int x = 0; x < img.cols; x++) {
            int b = ptr[x][0] / 32;  // Map 0-255 to 0-7
            int g = ptr[x][1] / 32;
            int r = ptr[x][2] / 32;

            // Convert 3D index to 1D
            int index = b + (g * bins) + (r * bins * bins);
            topHist.at<float>(0, index) += 1.0;
        }
    }

    // Process bottom half
    for(int y = midRow; y < img.rows; y++) {
        const Vec3b *ptr = img.ptr<Vec3b>(y);
        for(int x = 0; x < img.cols; x++) {
            int b = ptr[x][0] / 32;
            int g = ptr[x][1] / 32;
            int r = ptr[x][2] / 32;

            int index = b + (g * bins) + (r * bins * bins);
            bottomHist.at<float>(0, index) += 1.0;
        }
    }

    // Normalize each half
    topHist /= (midRow * img.cols);
    bottomHist /= ((img.rows - midRow) * img.cols);

    // Combine both histograms
    for(int i = 0; i < histSize; i++) {
        features.push_back(topHist.at<float>(0, i));
    }
    for(int i = 0; i < histSize; i++) {
        features.push_back(bottomHist.at<float>(0, i));
    }

    return features;
}

// Multi-histogram distance with equal weighting
float multiHistogramDistance(const vector<float> &h1, const vector<float> &h2) {
    if(h1.size() != h2.size()) {
        cerr << "Error: Feature vectors must be same size!" << endl;
        return -1.0;
    }

    int halfSize = h1.size() / 2;

    // Split into top and bottom
    vector<float> h1_top(h1.begin(), h1.begin() + halfSize);
    vector<float> h1_bottom(h1.begin() + halfSize, h1.end());
    vector<float> h2_top(h2.begin(), h2.begin() + halfSize);
    vector<float> h2_bottom(h2.begin() + halfSize, h2.end());

    // Compute intersection for each half
    float topIntersection = 0.0;
    for(int i = 0; i < halfSize; i++) {
        topIntersection += min(h1_top[i], h2_top[i]);
    }

    float bottomIntersection = 0.0;
    for(int i = 0; i < halfSize; i++) {
        bottomIntersection += min(h1_bottom[i], h2_bottom[i]);
    }

    // Convert to distances and average
    float topDistance = 1.0 - topIntersection;
    float bottomDistance = 1.0 - bottomIntersection;
    float distance = 0.5 * topDistance + 0.5 * bottomDistance;

    return distance;
}

// ==================== SOBEL FILTERS ====================

// Sobel X filter - detects vertical edges
// Uses separable filtering: [1 2 1] vertical, [-1 0 1] horizontal
int sobelX3x3(const Mat &src, Mat &dst){
    dst.create(src.size(), CV_16SC3);
    Mat temp(src.size(), CV_16SC3);
    
    // First pass: vertical smoothing
    for(int i=1; i<src.rows-1; i++){
        const Vec3b *sptr_up = src.ptr<Vec3b>(i-1);
        const Vec3b *sptr_mid = src.ptr<Vec3b>(i);
        const Vec3b *sptr_down = src.ptr<Vec3b>(i+1);
        Vec3s *tptr = temp.ptr<Vec3s>(i);
        
        for(int j=0; j<src.cols; j++){
            for(int c=0; c<3; c++){
                int value = sptr_up[j][c] + 2*sptr_mid[j][c] + sptr_down[j][c];
                tptr[j][c] = value;
            }
        }
    }
    
    // Second pass: horizontal derivative
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

// Sobel Y filter - detects horizontal edges
// Uses separable filtering: [1 2 1] horizontal, [1 0 -1] vertical
int sobelY3x3(const Mat &src, Mat &dst){
    dst.create(src.size(), CV_16SC3);
    Mat temp(src.size(), CV_16SC3);
    
    // First pass: horizontal smoothing
    for (int i = 0; i < src.rows; i++) {
        const Vec3b *sptr = src.ptr<Vec3b>(i);
        Vec3s *tptr = temp.ptr<Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                tptr[j][c] = sptr[j-1][c] + 2 * sptr[j][c] + sptr[j+1][c];
            }
        }
    }

    // Second pass: vertical derivative
    for (int i = 1; i < src.rows - 1; i++) {
        const Vec3s *up   = temp.ptr<Vec3s>(i-1);
        const Vec3s *down = temp.ptr<Vec3s>(i+1);
        Vec3s *dptr = dst.ptr<Vec3s>(i);
        
        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                dptr[j][c] = up[j][c] - down[j][c];
            }
        }
    }
    return 0;
}

// Compute gradient magnitude: sqrt(sobelX² + sobelY²)
int magnitude(const Mat &sobelX, const Mat &sobelY, Mat &dst){
    dst.create(sobelX.size(), CV_8UC3);
    
    for(int i=0; i<sobelX.rows; i++){
        const Vec3s *sxptr = sobelX.ptr<Vec3s>(i);
        const Vec3s *syptr = sobelY.ptr<Vec3s>(i);
        Vec3b *dptr = dst.ptr<Vec3b>(i);
        
        for(int j=0; j<sobelX.cols; j++){
            for(int c=0; c<3; c++){
                int value = sqrt(sxptr[j][c] * sxptr[j][c] + syptr[j][c] * syptr[j][c]);
                if(value>255) value = 255;
                dptr[j][c] = (uchar)value;
            }
        }
    }
    return 0;
}

// ==================== TEXTURE + COLOR FEATURES ====================

// Extract color histogram and texture histogram
// Returns 528 values (512 color + 16 texture)
vector<float> extractTextureColorFeatures(const Mat &img){
    vector<float> features;
    
    // Build color histogram (8×8×8 = 512 bins)
    const int bins = 8;
    const int histSize = bins*bins*bins;
    Mat colorHist = Mat::zeros(1, histSize, CV_32FC1);
    
    for(int y=0; y<img.rows;y++){
        const Vec3b *ptr = img.ptr<Vec3b>(y);
        for(int x=0; x<img.cols; x++){
            int b = ptr[x][0] /32;
            int g = ptr[x][1] /32;
            int r = ptr[x][2] /32;
            int index = b + (g*bins) + (r*bins*bins);
            colorHist.at<float>(0, index) += 1.0;
        }
    }
    colorHist /= (img.cols*img.rows);

    // Compute gradient magnitude for texture
    Mat sobelx, sobely;
    sobelX3x3(img, sobelx);
    sobelY3x3(img, sobely);
    Mat mag;
    magnitude(sobelx, sobely, mag);

    // Build texture histogram (16 bins of gradient magnitudes)
    const int textureBins = 16;
    Mat textureHist = Mat::zeros(1, textureBins, CV_32FC1);
    
    for(int y=0; y<mag.rows; y++){
        const Vec3b *ptr = mag.ptr<Vec3b>(y);
        for(int x=0; x<mag.cols; x++){
            int avgMag = (ptr[x][0] + ptr[x][1] + ptr[x][2]) /3;
            int bin = avgMag * textureBins / 256;
            if(bin >= textureBins) bin = textureBins - 1;
            textureHist.at<float>(0,bin) += 1.0;
        }
    }
    textureHist /= (mag.cols * mag.rows);
    
    // Combine both histograms
    for(int i=0; i<histSize; i++){
        features.push_back(colorHist.at<float>(0,i));
    }
    for(int i = 0; i < textureBins; i++) {
        features.push_back(textureHist.at<float>(0, i));
    }
    
    return features;
}

// Distance metric with 50/50 weighting
float textureColorDistance(const vector<float> &h1, const vector<float> &h2) {
    if(h1.size() != h2.size()) {
        cerr << "Error: Feature vectors must be same size!" << endl;
        return -1.0;
    }
    
    const int colorHistSize = 512;
    const int textureHistSize = 16;
    
    // Color histogram intersection
    vector<float> h1_color(h1.begin(), h1.begin() + colorHistSize);
    vector<float> h2_color(h2.begin(), h2.begin() + colorHistSize);
    
    float colorIntersection = 0.0;
    for(int i = 0; i < colorHistSize; i++) {
        colorIntersection += min(h1_color[i], h2_color[i]);
    }
    float colorDistance = 1.0 - colorIntersection;
    
    // Texture histogram intersection
    vector<float> h1_texture(h1.begin() + colorHistSize, h1.end());
    vector<float> h2_texture(h2.begin() + colorHistSize, h2.end());
    
    float textureIntersection = 0.0;
    for(int i = 0; i < textureHistSize; i++) {
        textureIntersection += min(h1_texture[i], h2_texture[i]);
    }
    float textureDistance = 1.0 - textureIntersection;
    
    // Combine with equal weights
    float finalDistance = 0.5 * colorDistance + 0.5 * textureDistance;
    return finalDistance;
}

// ==================== DEEP LEARNING EMBEDDINGS ====================

// Cosine distance: 1 - cos(angle between vectors)
// Better than Euclidean for high-dimensional embeddings
float cosineDistance(const vector<float> &v1, const vector<float> &v2){
    if(v1.size() != v2.size()){
        cout<< "Vectors must be same size!" << endl;
        return -1.0;
    }
    
    // Compute L2 norms
    float norm1 = 0.0;
    float norm2 = 0.0;
    for(size_t i=0; i<v1.size(); i++){
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);
    
    if(norm1 == 0 || norm2 == 0){
        return 2.0;  // Maximum distance
    }
    
    // Compute dot product of normalized vectors
    float dotProduct = 0.0;
    for(size_t i=0; i<v1.size(); i++){
        float n1 = v1[i]/norm1;
        float n2 = v2[i]/norm2;
        dotProduct += n1*n2;
    }
    
    // Clamp to avoid numerical errors
    dotProduct = max(-1.0f, min(1.0f, dotProduct));
    
    float distance = 1.0 - dotProduct;
    return distance;
}

// ==================== CUSTOM NATURE SCENE FEATURES ====================

// Extract features optimized for outdoor landscapes
// Combines spatial color (top/mid/bottom), texture, and DNN
// Returns 1680 values (1536 + 16 + 128)
vector<float> extractCustomNatureFeatures(const Mat &img, const vector<float> &dnnEmbedding) {
    vector<float> features;
    
    const int colorBins = 8;
    const int colorHistSize = colorBins * colorBins * colorBins;
    
    // Divide into three vertical regions
    int topEnd = img.rows / 3;
    int midEnd = 2 * img.rows / 3;
    
    Mat topHist = Mat::zeros(1, colorHistSize, CV_32FC1);
    Mat midHist = Mat::zeros(1, colorHistSize, CV_32FC1);
    Mat botHist = Mat::zeros(1, colorHistSize, CV_32FC1);
    
    // Build spatial color histograms
    for(int y = 0; y < img.rows; y++) {
        const Vec3b *ptr = img.ptr<Vec3b>(y);
        Mat *currentHist;
        
        // Select which region
        if(y < topEnd) currentHist = &topHist;
        else if(y < midEnd) currentHist = &midHist;
        else currentHist = &botHist;
        
        for(int x = 0; x < img.cols; x++) {
            int b = ptr[x][0] / 32;
            int g = ptr[x][1] / 32;
            int r = ptr[x][2] / 32;
            int index = b + (g * colorBins) + (r * colorBins * colorBins);
            currentHist->at<float>(0, index) += 1.0;
        }
    }
    
    // Normalize each region
    topHist /= (topEnd * img.cols);
    midHist /= ((midEnd - topEnd) * img.cols);
    botHist /= ((img.rows - midEnd) * img.cols);
    
    // Add spatial color to features
    for(int i = 0; i < colorHistSize; i++) {
        features.push_back(topHist.at<float>(0, i));
    }
    for(int i = 0; i < colorHistSize; i++) {
        features.push_back(midHist.at<float>(0, i));
    }
    for(int i = 0; i < colorHistSize; i++) {
        features.push_back(botHist.at<float>(0, i));
    }
    
    // Add texture histogram
    Mat sobelX, sobelY;
    sobelX3x3(img, sobelX);
    sobelY3x3(img, sobelY);
    Mat mag;
    magnitude(sobelX, sobelY, mag);
    
    const int textureBins = 16;
    Mat textureHist = Mat::zeros(1, textureBins, CV_32FC1);
    
    for(int y = 0; y < mag.rows; y++) {
        const Vec3b *ptr = mag.ptr<Vec3b>(y);
        for(int x = 0; x < mag.cols; x++) {
            int avgMag = (ptr[x][0] + ptr[x][1] + ptr[x][2]) / 3;
            int bin = avgMag * textureBins / 256;
            if(bin >= textureBins) bin = textureBins - 1;
            textureHist.at<float>(0, bin) += 1.0;
        }
    }
    textureHist /= (mag.rows * mag.cols);
    
    for(int i = 0; i < textureBins; i++) {
        features.push_back(textureHist.at<float>(0, i));
    }
    
    // Add downsampled DNN embeddings (every 4th value)
    for(size_t i = 0; i < dnnEmbedding.size(); i += 4) {
        features.push_back(dnnEmbedding[i]);
    }
    
    return features;
}

// Custom distance: 40% spatial color + 10% texture + 50% DNN
float customNatureDistance(const vector<float> &f1, const vector<float> &f2) {
    if(f1.size() != f2.size()) {
        cerr << "Error: Feature vectors must be same size!" << endl;
        return -1.0;
    }
    
    const int colorHistSize = 512;
    const int textureBins = 16;
    
    // Color distance (average over 3 regions)
    float colorDist = 0.0;
    for(int region = 0; region < 3; region++) {
        int offset = region * colorHistSize;
        float intersection = 0.0;
        for(int i = 0; i < colorHistSize; i++) {
            intersection += min(f1[offset + i], f2[offset + i]);
        }
        colorDist += (1.0 - intersection);
    }
    colorDist /= 3.0;
    
    // Texture distance
    int textureOffset = 3 * colorHistSize;
    float textureIntersection = 0.0;
    for(int i = 0; i < textureBins; i++) {
        textureIntersection += min(f1[textureOffset + i], f2[textureOffset + i]);
    }
    float textureDist = 1.0 - textureIntersection;
    
    // DNN distance
    int dnnOffset = textureOffset + textureBins;
    vector<float> dnn1(f1.begin() + dnnOffset, f1.end());
    vector<float> dnn2(f2.begin() + dnnOffset, f2.end());
    float dnnDist = cosineDistance(dnn1, dnn2);
    
    // Weighted combination
    float finalDist = 0.4 * colorDist + 0.1 * textureDist + 0.5 * dnnDist;
    return finalDist;
}

// ==================== ADAPTIVE FACE-AWARE FEATURES ====================

// Detect faces with strict validation to filter false positives
int detectFaces(const Mat &img, vector<Rect> &faces) {
    static CascadeClassifier face_cascade;
    static bool cascade_loaded = false;
    
    // Load cascade classifier once
    if(!cascade_loaded) {
        vector<string> paths = {
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_default.xml",
            "./haarcascade_frontalface_alt2.xml"
        };
        
        for(const auto &path : paths) {
            if(face_cascade.load(path)) {
                cascade_loaded = true;
                break;
            }
        }
        
        if(!cascade_loaded) {
            return 0;
        }
    }
    
    // Prepare image for detection
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    
    // Detect with strict parameters
    vector<Rect> detected_faces;
    face_cascade.detectMultiScale(gray, detected_faces, 1.1, 5, 0, Size(80, 80));
    
    // Filter by size and aspect ratio
    for(const auto &face : detected_faces) {
        float faceArea = face.width * face.height;
        float imageArea = img.rows * img.cols;
        float areaRatio = faceArea / imageArea;
        
        // Face should be 5-70% of image
        if(areaRatio > 0.05 && areaRatio < 0.70) {
            float aspectRatio = (float)face.width / face.height;
            // Face should be roughly square
            if(aspectRatio > 0.7 && aspectRatio < 1.4) {
                faces.push_back(face);
            }
        }
    }
    
    return faces.size();
}

// Extract face features with skin tone analysis
// Returns 513 values (1 skin ratio + 256 face hist + 256 skin hist)
vector<float> extractFaceFeatures(const Mat &img, const vector<Rect> &faces) {
    const int bins = 8;
    const int histSize = bins * bins * bins;
    
    Mat faceHist = Mat::zeros(1, histSize, CV_32FC1);
    Mat skinToneHist = Mat::zeros(1, histSize, CV_32FC1);
    int totalPixels = 0;
    int skinPixels = 0;
    
    // Process each detected face
    for(const auto &face : faces) {
        Rect safeFace = face & Rect(0, 0, img.cols, img.rows);
        if(safeFace.width == 0 || safeFace.height == 0) continue;
        
        Mat faceROI = img(safeFace);
        
        for(int y = 0; y < faceROI.rows; y++) {
            const Vec3b *ptr = faceROI.ptr<Vec3b>(y);
            for(int x = 0; x < faceROI.cols; x++) {
                int b = ptr[x][0];
                int g = ptr[x][1];
                int r = ptr[x][2];
                
                int bBin = b / 32;
                int gBin = g / 32;
                int rBin = r / 32;
                int index = bBin + (gBin * bins) + (rBin * bins * bins);
                
                faceHist.at<float>(0, index) += 1.0;
                totalPixels++;
                
                // Check for skin tone (R > G > B in specific ranges)
                if(r > 95 && g > 40 && b > 20 &&
                   r > g && r > b && abs(r - g) > 15) {
                    skinToneHist.at<float>(0, index) += 1.0;
                    skinPixels++;
                }
            }
        }
    }
    
    // Normalize histograms
    if(totalPixels > 0) faceHist /= totalPixels;
    if(skinPixels > 0) skinToneHist /= skinPixels;
    
    // Combine into feature vector
    vector<float> features;
    float skinRatio = (totalPixels > 0) ? (float)skinPixels / totalPixels : 0.0;
    features.push_back(skinRatio);
    
    // Subsample histograms to 256 values each
    for(int i = 0; i < histSize; i += 2) {
        features.push_back(faceHist.at<float>(0, i));
    }
    for(int i = 0; i < histSize; i += 2) {
        features.push_back(skinToneHist.at<float>(0, i));
    }
    
    return features;
}

// Extract adaptive features based on face presence
// Returns 1794 values (1 + 513 + 1024 + 256)
vector<float> extractAdaptiveFeatures(const Mat &img, const vector<float> &dnnEmbedding) {
    vector<float> features;
    
    // Detect faces
    vector<Rect> faces;
    int numFaces = detectFaces(img, faces);
    features.push_back((float)numFaces);
    
    // Extract face features if present
    if(numFaces > 0) {
        vector<float> faceFeats = extractFaceFeatures(img, faces);
        features.insert(features.end(), faceFeats.begin(), faceFeats.end());
    } else {
        // Add zeros as placeholders
        for(int i = 0; i < 513; i++) {
            features.push_back(0.0);
        }
    }
    
    // Add spatial color histograms (top/bottom)
    const int bins = 8;
    const int histSize = bins * bins * bins;
    
    Mat topHist = Mat::zeros(1, histSize, CV_32FC1);
    Mat botHist = Mat::zeros(1, histSize, CV_32FC1);
    int midRow = img.rows / 2;
    
    for(int y = 0; y < img.rows; y++) {
        const Vec3b *ptr = img.ptr<Vec3b>(y);
        Mat *currentHist = (y < midRow) ? &topHist : &botHist;
        
        for(int x = 0; x < img.cols; x++) {
            int b = ptr[x][0] / 32;
            int g = ptr[x][1] / 32;
            int r = ptr[x][2] / 32;
            int index = b + (g * bins) + (r * bins * bins);
            currentHist->at<float>(0, index) += 1.0;
        }
    }
    
    topHist /= (midRow * img.cols);
    botHist /= ((img.rows - midRow) * img.cols);
    
    for(int i = 0; i < histSize; i++) {
        features.push_back(topHist.at<float>(0, i));
    }
    for(int i = 0; i < histSize; i++) {
        features.push_back(botHist.at<float>(0, i));
    }
    
    // Add downsampled DNN embeddings
    for(size_t i = 0; i < dnnEmbedding.size(); i += 2) {
        features.push_back(dnnEmbedding[i]);
    }
    
    return features;
}

// Adaptive distance with conditional weighting based on face presence
float adaptiveDistance(const vector<float> &f1, const vector<float> &f2) {
    if(f1.size() != f2.size()) {
        cerr << "Error: Feature vectors must be same size!" << endl;
        return -1.0;
    }
    
    // Extract face counts
    int target_faces = (int)f1[0];
    int query_faces = (int)f2[0];
    bool target_has_face = (target_faces > 0);
    bool query_has_face = (query_faces > 0);
    
    // Define offsets in feature vector
    int faceOffset = 1;
    int spatialColorOffset = faceOffset + 513;
    int dnnOffset = spatialColorOffset + 1024;
    
    float faceDist = 0.0;
    float spatialColorDist = 0.0;
    float dnnDist = 0.0;
    
    // Compute face distance if both have faces
    if(target_has_face && query_has_face) {
        float target_skin = f1[faceOffset];
        float query_skin = f2[faceOffset];
        
        // Check if likely human faces (skin ratio > 0.2)
        if(target_skin > 0.2 && query_skin > 0.2) {
            // Compare face and skin histograms
            float faceHistIntersection = 0.0;
            float skinHistIntersection = 0.0;
            
            for(int i = 1; i <= 256; i++) {
                faceHistIntersection += min(f1[faceOffset + i], f2[faceOffset + i]);
            }
            for(int i = 257; i <= 512; i++) {
                skinHistIntersection += min(f1[faceOffset + i], f2[faceOffset + i]);
            }
            
            // Weight skin tone more heavily
            faceDist = 0.3 * (1.0 - faceHistIntersection) + 0.7 * (1.0 - skinHistIntersection);
        } else {
            // Possible false positives (animals, objects)
            faceDist = 0.6;
        }
    } else if(target_has_face != query_has_face) {
        // Type mismatch penalty
        faceDist = 0.8;
    }
    
    // Compute spatial color distance
    float topIntersection = 0.0;
    float botIntersection = 0.0;
    
    for(int i = 0; i < 512; i++) {
        topIntersection += min(f1[spatialColorOffset + i], f2[spatialColorOffset + i]);
    }
    for(int i = 512; i < 1024; i++) {
        botIntersection += min(f1[spatialColorOffset + i], f2[spatialColorOffset + i]);
    }
    
    spatialColorDist = 0.5 * (1.0 - topIntersection) + 0.5 * (1.0 - botIntersection);
    
    // Compute DNN distance
    vector<float> dnn1(f1.begin() + dnnOffset, f1.end());
    vector<float> dnn2(f2.begin() + dnnOffset, f2.end());
    dnnDist = cosineDistance(dnn1, dnn2);
    
    // Adaptive weighting based on face presence
    float finalDist;
    
    if(target_has_face && query_has_face) {
        float target_skin = f1[faceOffset];
        float query_skin = f2[faceOffset];
        
        if(target_skin > 0.2 && query_skin > 0.2) {
            // Both human faces: prioritize face features
            finalDist = 0.50 * faceDist + 0.10 * spatialColorDist + 0.40 * dnnDist;
        } else {
            // Possible false positives: rely on DNN
            finalDist = 0.20 * faceDist + 0.20 * spatialColorDist + 0.60 * dnnDist;
        }
    } else if(!target_has_face && !query_has_face) {
        // No faces: use spatial color and DNN
        finalDist = 0.40 * spatialColorDist + 0.60 * dnnDist;
    } else {
        // Type mismatch: heavy penalty
        finalDist = 0.30 * faceDist + 0.20 * spatialColorDist + 0.50 * dnnDist;
    }
    
    return finalDist;
}

// ==================== TRASH CAN SPECIALIZED FEATURES ====================

// Extract features optimized for blue trash bins
// Returns 1810 values (1 + 16 + 1 + 1536 + 256)
vector<float> extractTrashCanFeatures(const Mat &img, const vector<float> &dnnEmbedding) {
    vector<float> features;
    
    // Blue color analysis
    const int colorBins = 16;
    Mat blueHist = Mat::zeros(1, colorBins, CV_32FC1);
    int bluePixelCount = 0;
    int totalPixels = img.rows * img.cols;
    
    // Focus on bottom 2/3 where trash cans usually appear
    int startRow = img.rows / 3;
    
    for(int y = startRow; y < img.rows; y++) {
        const Vec3b *ptr = img.ptr<Vec3b>(y);
        for(int x = 0; x < img.cols; x++) {
            int b = ptr[x][0];
            int g = ptr[x][1];
            int r = ptr[x][2];
            
            int blueBin = b * colorBins / 256;
            if(blueBin >= colorBins) blueBin = colorBins - 1;
            blueHist.at<float>(0, blueBin) += 1.0;
            
            // Count strong blue pixels
            if(b > r + 20 && b > g + 20 && b > 80) {
                bluePixelCount++;
            }
        }
    }
    
    int regionPixels = (img.rows - startRow) * img.cols;
    blueHist /= regionPixels;
    
    float blueRatio = (float)bluePixelCount / totalPixels;
    features.push_back(blueRatio);
    
    for(int i = 0; i < colorBins; i++) {
        features.push_back(blueHist.at<float>(0, i));
    }
    
    // Vertical edge detection (trash cans are rectangular)
    Mat sobelX, sobelY;
    sobelX3x3(img, sobelX);
    sobelY3x3(img, sobelY);
    
    int verticalEdges = 0;
    int horizontalEdges = 0;
    
    for(int y = 1; y < sobelX.rows - 1; y++) {
        const Vec3s *sxptr = sobelX.ptr<Vec3s>(y);
        const Vec3s *syptr = sobelY.ptr<Vec3s>(y);
        
        for(int x = 1; x < sobelX.cols - 1; x++) {
            int gx = abs(sxptr[x][0]) + abs(sxptr[x][1]) + abs(sxptr[x][2]);
            int gy = abs(syptr[x][0]) + abs(syptr[x][1]) + abs(syptr[x][2]);
            
            if(gx > 100 || gy > 100) {
                if(gy > gx * 1.5) verticalEdges++;
                else if(gx > gy * 1.5) horizontalEdges++;
            }
        }
    }
    
    float verticalRatio = 0.0;
    if(verticalEdges + horizontalEdges > 0) {
        verticalRatio = (float)verticalEdges / (verticalEdges + horizontalEdges);
    }
    features.push_back(verticalRatio);
    
    // Spatial color histograms (3 vertical regions)
    const int bins = 8;
    const int histSize = bins * bins * bins;
    
    Mat topHist = Mat::zeros(1, histSize, CV_32FC1);
    Mat midHist = Mat::zeros(1, histSize, CV_32FC1);
    Mat botHist = Mat::zeros(1, histSize, CV_32FC1);
    
    int topEnd = img.rows / 3;
    int midEnd = 2 * img.rows / 3;
    
    for(int y = 0; y < img.rows; y++) {
        const Vec3b *ptr = img.ptr<Vec3b>(y);
        Mat *currentHist;
        
        if(y < topEnd) currentHist = &topHist;
        else if(y < midEnd) currentHist = &midHist;
        else currentHist = &botHist;
        
        for(int x = 0; x < img.cols; x++) {
            int bBin = ptr[x][0] / 32;
            int gBin = ptr[x][1] / 32;
            int rBin = ptr[x][2] / 32;
            int index = bBin + (gBin * bins) + (rBin * bins * bins);
            currentHist->at<float>(0, index) += 1.0;
        }
    }
    
    topHist /= (topEnd * img.cols);
    midHist /= ((midEnd - topEnd) * img.cols);
    botHist /= ((img.rows - midEnd) * img.cols);
    
    for(int i = 0; i < histSize; i++) {
        features.push_back(topHist.at<float>(0, i));
    }
    for(int i = 0; i < histSize; i++) {
        features.push_back(midHist.at<float>(0, i));
    }
    for(int i = 0; i < histSize; i++) {
        features.push_back(botHist.at<float>(0, i));
    }
    
    // Add downsampled DNN embeddings
    for(size_t i = 0; i < dnnEmbedding.size(); i += 2) {
        features.push_back(dnnEmbedding[i]);
    }
    
    return features;
}

// Distance metric optimized for trash can matching
// Heavily weights blue color (30% total)
float trashCanDistance(const vector<float> &f1, const vector<float> &f2) {
    if(f1.size() != f2.size()) {
        cerr << "Error: Feature vectors must be same size!" << endl;
        return -1.0;
    }
    
    // Extract specialized features
    float blueRatio1 = f1[0];
    float blueRatio2 = f2[0];
    float vertRatio1 = f1[17];
    float vertRatio2 = f2[17];
    
    // Blue ratio distance
    float blueDist = abs(blueRatio1 - blueRatio2);
    
    // Vertical ratio distance
    float vertDist = abs(vertRatio1 - vertRatio2);
    
    // Blue histogram distance
    float blueHistIntersection = 0.0;
    for(int i = 1; i <= 16; i++) {
        blueHistIntersection += min(f1[i], f2[i]);
    }
    float blueHistDist = 1.0 - blueHistIntersection;
    
    // Spatial color distance
    float spatialDist = 0.0;
    int spatialOffset = 18;
    
    for(int region = 0; region < 3; region++) {
        float intersection = 0.0;
        int offset = spatialOffset + (region * 512);
        
        for(int i = 0; i < 512; i++) {
            intersection += min(f1[offset + i], f2[offset + i]);
        }
        spatialDist += (1.0 - intersection);
    }
    spatialDist /= 3.0;
    
    // DNN distance
    int dnnOffset = spatialOffset + 1536;
    vector<float> dnn1(f1.begin() + dnnOffset, f1.end());
    vector<float> dnn2(f2.begin() + dnnOffset, f2.end());
    float dnnDist = cosineDistance(dnn1, dnn2);
    
    // Weighted combination (blue color is most important)
    float finalDist = 0.20 * blueDist + 
                      0.10 * blueHistDist + 
                      0.05 * vertDist + 
                      0.15 * spatialDist + 
                      0.50 * dnnDist;
    
    return finalDist;
}