/*
  Astitva Goel
  January 2026
  CS 5330 Computer Vision

  Header file containing function prototypes for image filtering operations
  and video effects including edge detection, blurring, face-based effects etc.
*/
#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

int grayscale(cv::Mat &src, cv::Mat &dst);
int sepiatone(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &dst);
int quantize(cv::Mat &src, cv::Mat &dst, int levels);
int blurBackgroundExceptFace(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);
int erodeVid(cv::Mat &src, cv::Mat &dst);
int embossingEffect(cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &dst);
int grayBackgroundExceptFace(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);
void addCaptionToFrame(cv::Mat &frame, const std::string& topText, const std::string& bottomText);
std::string getCaptionFromUser(const std::string& prompt);

#endif