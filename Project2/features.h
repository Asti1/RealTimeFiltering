/*
  Astitva Goel
  Feature extraction and distance metric functions
*/
#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <opencv2/objdetect.hpp> 
// Extract 7x7 center region as feature vector (Baseline)
std::vector<float> extractBaselineFeatures(const cv::Mat &img);

// Extract RG chromaticity histogram
std::vector<float> extractColorHistogramFeatures(const cv::Mat &img);

// Extract multi-histogram (top and bottom halves) using RGB
std::vector<float> extractMultiHistogramFeatures(const cv::Mat &img);

// Compute Sum of Squared Differences between two feature vectors
float computeSSD(const std::vector<float> &f1, const std::vector<float> &f2);

// Histogram intersection distance
float histogramIntersection(const std::vector<float> &h1, const std::vector<float> &h2);

// Multi-histogram distance (for top/bottom histograms)
float multiHistogramDistance(const std::vector<float> &h1, const std::vector<float> &h2);

// Texture + Color features
std::vector<float> extractTextureColorFeatures(const cv::Mat &img);
float textureColorDistance(const std::vector<float> &h1, const std::vector<float> &h2);
int sobelY3x3(const cv::Mat &src, cv::Mat &dst);
int sobelX3x3(const cv::Mat &src, cv::Mat &dst);
int magnitude(const cv::Mat &sobelX, const cv::Mat &sobelY, cv::Mat &dst);
// Deep network embedding distance
float cosineDistance(const std::vector<float> &v1, const std::vector<float> &v2);
// Custom feature extraction for nature scenes
std::vector<float> extractCustomNatureFeatures(const cv::Mat &img, const std::vector<float> &dnnEmbedding);
float customNatureDistance(const std::vector<float> &f1, const std::vector<float> &f2);




// Adaptive face-aware features
int detectFaces(const cv::Mat &img, std::vector<cv::Rect> &faces);
std::vector<float> extractFaceFeatures(const cv::Mat &img, const std::vector<cv::Rect> &faces);
std::vector<float> extractAdaptiveFeatures(const cv::Mat &img, const std::vector<float> &dnnEmbedding);
float adaptiveDistance(const std::vector<float> &f1, const std::vector<float> &f2);
// Trash can specific features
std::vector<float> extractTrashCanFeatures(const cv::Mat &img, const std::vector<float> &dnnEmbedding);
float trashCanDistance(const std::vector<float> &f1, const std::vector<float> &f2);

#endif