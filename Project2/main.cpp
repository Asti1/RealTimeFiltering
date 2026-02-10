/*
  Astitva Goel
  Content-Based Image Retrieval
  Main program for image matching
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <dirent.h>
#include <cstring>
#include "features.h"
#include "csv_util.h"

using namespace cv;
using namespace std;

// Structure to hold image filename and its distance from target
struct ImageMatch {
    string filename;
    float distance;
    // Constructor
    ImageMatch(string name, float dist) : filename(name), distance(dist) {}
};
// Helper function to trim whitespace from a C string
void trim(char* str) {
    // Trim leading spaces
    int start = 0;
    while(str[start] == ' ' || str[start] == '\t' || str[start] == '\r' || str[start] == '\n') {
        start++;
    }
    
    // Shift string left if needed
    if(start > 0) {
        int i = 0;
        while(str[start + i] != '\0') {
            str[i] = str[start + i];
            i++;
        }
        str[i] = '\0';
    }
    
    // Trim trailing spaces
    int end = strlen(str) - 1;
    while(end >= 0 && (str[end] == ' ' || str[end] == '\t' || str[end] == '\r' || str[end] == '\n')) {
        str[end] = '\0';
        end--;
    }
}

int main(int argc, char *argv[]) {
    // Check for correct number of arguments
    if(argc < 5) {
        cout << "Usage: " << argv[0] << " <target_image> <image_directory> <method> <N_matches>" << endl;
        cout << "Methods: baseline, histogram, multihist" << endl;
        return -1;
    }

    // Parse command line arguments
    string targetFile = argv[1];
    string dirname = argv[2];
    string method = argv[3];
    int N = atoi(argv[4]);

    cout << "Target image: " << targetFile << endl;
    cout << "Image directory: " << dirname << endl;
    cout << "Method: " << method << endl;
    cout << "Number of matches to return: " << N << endl;
    // ========== HANDLE EMBEDDING METHOD ==========
    if(method == "embedding") {
        if(argc < 6) {
            cout << "Error: Embedding method requires CSV file path" << endl;
            cout << "Usage: " << argv[0] << " <target> <dir> embedding <N> <csv_file>" << endl;
            return -1;
        }
        
        char *csvPath = argv[5];
        cout << "CSV file: " << csvPath << endl << endl;
        
        // Load embeddings using existing CSV utility
        vector<char *> filenames;
        vector<vector<float>> data;
        
        if(read_image_data_csv(csvPath, filenames, data, 0) != 0) {
            cout << "Error: Failed to read CSV file" << endl;
            return -1;
        }
        
        // TRIM WHITESPACE FROM ALL FILENAMES
        for(size_t i = 0; i < filenames.size(); i++) {
            trim(filenames[i]);
        }
        
        cout << "Loaded " << filenames.size() << " embeddings" << endl;
        
        // Extract just the filename from target path
        string targetFilename = targetFile;
        size_t lastSlash = targetFilename.find_last_of("/\\");
        if(lastSlash != string::npos) {
            targetFilename = targetFilename.substr(lastSlash + 1);
        }
        
        cout << "Looking for target: '" << targetFilename << "'" << endl;
        
        // Find target embedding
        int targetIndex = -1;
        for(size_t i = 0; i < filenames.size(); i++) {
            // Debug: print comparison for first few
            if(i < 5) {
                cout << "Comparing '" << filenames[i] << "' with '" << targetFilename << "'" << endl;
            }
            
            if(strcmp(filenames[i], targetFilename.c_str()) == 0) {
                targetIndex = i;
                cout << "Found match at index " << i << endl;
                break;
            }
        }
        
        if(targetIndex == -1) {
            cout << "Error: Target image '" << targetFilename << "' not found in CSV" << endl;
            cout << "\nSearching for partial matches..." << endl;
            
            // Try to find partial matches for debugging
            for(size_t i = 0; i < filenames.size(); i++) {
                if(strstr(filenames[i], "0164") != NULL) {
                    cout << "Found file with '0164': '" << filenames[i] << "'" << endl;
                }
            }
            
            // Print available filenames for debugging
            cout << "\nAvailable images (first 10):" << endl;
            for(size_t i = 0; i < min((size_t)10, filenames.size()); i++) {
                cout << "  '" << filenames[i] << "' (length: " << strlen(filenames[i]) << ")" << endl;
            }
            return -1;
        }
        
        vector<float> targetFeatures = data[targetIndex];
        cout << "Target embedding loaded (" << targetFeatures.size() << " values)" << endl << endl;
        
        // Compute distances to all images
        vector<ImageMatch> matches;
        
        for(size_t i = 0; i < filenames.size(); i++) {
            // Compute cosine distance
            float distance = cosineDistance(targetFeatures, data[i]);
            matches.push_back(ImageMatch(string(filenames[i]), distance));
        }
        
        // Sort by distance
        sort(matches.begin(), matches.end(),
             [](const ImageMatch &a, const ImageMatch &b) {
                 return a.distance < b.distance;
             });
        
        // Display results
        cout << "Top " << N << " matches:" << endl;
        cout << "========================================" << endl;
        for(int i = 0; i < N && i < static_cast<int>(matches.size()); i++) {
            cout << (i+1) << ". " << matches[i].filename
                 << " (distance: " << matches[i].distance << ")" << endl;
        }
        
        // Clean up
        for(size_t i = 0; i < filenames.size(); i++) {
            delete[] filenames[i];
        }
        
        return 0;
    }
    if(method == "custom") {
        if(argc < 6) {
            cout << "Error: Custom method requires CSV file path" << endl;
            cout << "Usage: " << argv[0] << " <target> <dir> custom <N> <csv_file>" << endl;
            return -1;
        }
        
        char *csvPath = argv[5];
        cout << "CSV file: " << csvPath << endl << endl;
        
        // Load DNN embeddings
        vector<char *> filenames;
        vector<vector<float>> dnnData;
        
        if(read_image_data_csv(csvPath, filenames, dnnData, 0) != 0) {
            cout << "Error: Failed to read CSV file" << endl;
            return -1;
        }
        
        // Trim whitespace
        for(size_t i = 0; i < filenames.size(); i++) {
            trim(filenames[i]);
        }
        
        cout << "Loaded " << filenames.size() << " embeddings" << endl;
        
        // Get target filename
        string targetFilename = targetFile;
        size_t lastSlash = targetFilename.find_last_of("/\\");
        if(lastSlash != string::npos) {
            targetFilename = targetFilename.substr(lastSlash + 1);
        }
        
        // Find target in CSV
        int targetIndex = -1;
        for(size_t i = 0; i < filenames.size(); i++) {
            if(strcmp(filenames[i], targetFilename.c_str()) == 0) {
                targetIndex = i;
                break;
            }
        }
        
        if(targetIndex == -1) {
            cout << "Error: Target not found in CSV" << endl;
            return -1;
        }
        
        // Load target image and extract custom features
        Mat targetImg = imread(targetFile);
        if(targetImg.empty()) {
            cout << "Error: Could not read target image" << endl;
            return -1;
        }
        
        vector<float> targetFeatures = extractCustomNatureFeatures(targetImg, dnnData[targetIndex]);
        cout << "Target features extracted (" << targetFeatures.size() << " values)" << endl << endl;
        
        // Process all images
        cout << "Processing database images..." << endl;
        vector<ImageMatch> matches;
        int processed = 0;
        
        for(size_t i = 0; i < filenames.size(); i++) {
            // Build filepath
            string filepath = dirname + "/" + string(filenames[i]);
            
            Mat img = imread(filepath);
            if(img.empty()) {
                continue;
            }
            
            // Extract custom features
            vector<float> features = extractCustomNatureFeatures(img, dnnData[i]);
            
            // Compute distance
            float distance = customNatureDistance(targetFeatures, features);
            
            matches.push_back(ImageMatch(string(filenames[i]), distance));
            processed++;
            
            if(processed % 100 == 0) {
                cout << "  Processed " << processed << " images..." << endl;
            }
        }
        
        cout << "Processed " << processed << " images" << endl << endl;
        
        // Sort by distance
        sort(matches.begin(), matches.end(),
             [](const ImageMatch &a, const ImageMatch &b) {
                 return a.distance < b.distance;
             });
        
        // Display top N matches
        cout << "Top " << N << " matches:" << endl;
        cout << "========================================" << endl;
        for(int i = 0; i < N && i < static_cast<int>(matches.size()); i++) {
            cout << (i+1) << ". " << matches[i].filename
                 << " (distance: " << matches[i].distance << ")" << endl;
        }
        
        // Display bottom 5 (least similar)
        cout << "\nLeast similar (bottom 5):" << endl;
        cout << "========================================" << endl;
        int start = max(0, (int)matches.size() - 5);
        for(int i = start; i < matches.size(); i++) {
            cout << (matches.size() - i) << ". " << matches[i].filename
                 << " (distance: " << matches[i].distance << ")" << endl;
        }
        
        // Clean up
        for(size_t i = 0; i < filenames.size(); i++) {
            delete[] filenames[i];
        }
        
        return 0;
    }
    
   // ========== HANDLE ADAPTIVE METHOD ==========
    if(method == "adaptive") {
        if(argc < 6) {
            cout << "Error: Adaptive method requires CSV file path" << endl;
            return -1;
        }
        
        char *csvPath = argv[5];
        
        // Load DNN embeddings
        vector<char *> filenames;
        vector<vector<float>> dnnData;
        
        if(read_image_data_csv(csvPath, filenames, dnnData, 0) != 0) {
            cout << "Error: Failed to read CSV file" << endl;
            return -1;
        }
        
        for(size_t i = 0; i < filenames.size(); i++) {
            trim(filenames[i]);
        }
        
        // Get target
        string targetFilename = targetFile;
        size_t lastSlash = targetFilename.find_last_of("/\\");
        if(lastSlash != string::npos) {
            targetFilename = targetFilename.substr(lastSlash + 1);
        }
        
        int targetIndex = -1;
        for(size_t i = 0; i < filenames.size(); i++) {
            if(strcmp(filenames[i], targetFilename.c_str()) == 0) {
                targetIndex = i;
                break;
            }
        }
        
        if(targetIndex == -1) {
            cout << "Error: Target not found in CSV" << endl;
            return -1;
        }
        
        // Load target and extract adaptive features
        Mat targetImg = imread(targetFile);
        vector<float> targetFeatures = extractAdaptiveFeatures(targetImg, dnnData[targetIndex]);
        
        bool targetHasFace = (targetFeatures[0] > 0.5);
        cout << "Target image " << (targetHasFace ? "HAS FACE(S)" : "has no faces") << endl;
        cout << "Extracted features: " << targetFeatures.size() << " values" << endl << endl;
        
        // Process database
        cout << "Processing database..." << endl;
        vector<ImageMatch> matches;
        
        for(size_t i = 0; i < filenames.size(); i++) {
            string filepath = dirname + "/" + string(filenames[i]);
            Mat img = imread(filepath);
            
            if(img.empty()) continue;
            
            vector<float> features = extractAdaptiveFeatures(img, dnnData[i]);
            float distance = adaptiveDistance(targetFeatures, features);
            
            matches.push_back(ImageMatch(string(filenames[i]), distance));
        }
        
        // Sort and display
        sort(matches.begin(), matches.end(),
             [](const ImageMatch &a, const ImageMatch &b) {
                 return a.distance < b.distance;
             });
        
        cout << "\nTop " << N << " matches:" << endl;
        cout << "========================================" << endl;
        for(int i = 0; i < N && i < matches.size(); i++) {
            cout << (i+1) << ". " << matches[i].filename
                 << " (distance: " << matches[i].distance << ")" << endl;
        }
        
        cout << "\nLeast similar (bottom 5):" << endl;
        cout << "========================================" << endl;
        int start = max(0, (int)matches.size() - 5);
        for(int i = start; i < matches.size(); i++) {
            cout << (i - start + 1) << ". " << matches[i].filename
                 << " (distance: " << matches[i].distance << ")" << endl;
        }
        
        // Clean up
        for(size_t i = 0; i < filenames.size(); i++) {
            delete[] filenames[i];
        }
        
        return 0;
    }

    // ========== HANDLE TRASH CAN METHOD ==========
    if(method == "trashcan") {
        if(argc < 6) {
            cout << "Error: Trashcan method requires CSV file path" << endl;
            return -1;
        }
        
        char *csvPath = argv[5];
        
        // Load DNN embeddings
        vector<char *> filenames;
        vector<vector<float>> dnnData;
        
        if(read_image_data_csv(csvPath, filenames, dnnData, 0) != 0) {
            cout << "Error: Failed to read CSV file" << endl;
            return -1;
        }
        
        for(size_t i = 0; i < filenames.size(); i++) {
            trim(filenames[i]);
        }
        
        // Get target
        string targetFilename = targetFile;
        size_t lastSlash = targetFilename.find_last_of("/\\");
        if(lastSlash != string::npos) {
            targetFilename = targetFilename.substr(lastSlash + 1);
        }
        
        int targetIndex = -1;
        for(size_t i = 0; i < filenames.size(); i++) {
            if(strcmp(filenames[i], targetFilename.c_str()) == 0) {
                targetIndex = i;
                break;
            }
        }
        
        if(targetIndex == -1) {
            cout << "Error: Target not found in CSV" << endl;
            return -1;
        }
        
        // Load target and extract features
        Mat targetImg = imread(targetFile);
        vector<float> targetFeatures = extractTrashCanFeatures(targetImg, dnnData[targetIndex]);
        
        cout << "Target features extracted (" << targetFeatures.size() << " values)" << endl;
        cout << "Target blue ratio: " << targetFeatures[0] << endl;
        cout << "Target vertical edge ratio: " << targetFeatures[17] << endl << endl;
        
        // Process database
        cout << "Processing database..." << endl;
        vector<ImageMatch> matches;
        
        for(size_t i = 0; i < filenames.size(); i++) {
            string filepath = dirname + "/" + string(filenames[i]);
            Mat img = imread(filepath);
            
            if(img.empty()) continue;
            
            vector<float> features = extractTrashCanFeatures(img, dnnData[i]);
            float distance = trashCanDistance(targetFeatures, features);
            
            matches.push_back(ImageMatch(string(filenames[i]), distance));
        }
        
        // Sort by distance
        sort(matches.begin(), matches.end(),
             [](const ImageMatch &a, const ImageMatch &b) {
                 return a.distance < b.distance;
             });
        
        cout << "\n========================================" << endl;
        cout << "RECALL EVALUATION FOR TRASH CANS" << endl;
        cout << "========================================\n" << endl;
        
        // Display top N matches
        cout << "Top " << N << " matches:" << endl;
        cout << "----------------------------------------" << endl;
        for(int i = 0; i < N && i < matches.size(); i++) {
            cout << (i+1) << ". " << matches[i].filename
                 << " (distance: " << matches[i].distance << ")" << endl;
        }
        
        // Show extended results for recall calculation
        cout << "\nTop 20 matches (for recall analysis):" << endl;
        cout << "----------------------------------------" << endl;
        for(int i = 0; i < 20 && i < matches.size(); i++) {
            cout << (i+1) << ". " << matches[i].filename
                 << " (distance: " << matches[i].distance << ")" << endl;
        }

        // Clean up
        for(size_t i = 0; i < filenames.size(); i++) {
            delete[] filenames[i];
        }

        return 0;
    }
    // Read target image
    Mat targetImg = imread(targetFile);
    if(targetImg.empty()) {
        cout << "Could not read target image: " << targetFile << endl;
        return -1;
    }

    // Extract features based on method
    vector<float> targetFeatures;
    if(method == "baseline") {
        targetFeatures = extractBaselineFeatures(targetImg);
    } else if(method == "histogram") {
        targetFeatures = extractColorHistogramFeatures(targetImg);
    } else if(method == "multihist") {
        targetFeatures = extractMultiHistogramFeatures(targetImg);
    }else if(method == "texture") {
        targetFeatures = extractTextureColorFeatures(targetImg);
    } else {
        cout << "Unknown method: " << method << endl;
        return -1;
    }

    cout << "Extracted features from target image (" << targetFeatures.size() << " values)" << endl;

    // Process database
    vector<ImageMatch> matches;
    DIR *dirp;
    struct dirent *dp;

    dirp = opendir(dirname.c_str());
    if(dirp == NULL) {
        cout << "Cannot open directory " << dirname << endl;
        return -1;
    }

    cout << "Processing images in database" << endl;
    int imageCount = 0;

    while((dp = readdir(dirp)) != NULL) {
        if(strstr(dp->d_name, ".jpg") ||
           strstr(dp->d_name, ".png") ||
           strstr(dp->d_name, ".ppm") ||
           strstr(dp->d_name, ".tif")) {

            string filepath = dirname + "/" + dp->d_name;
            Mat img = imread(filepath);
            
            if(img.empty()) {
                continue;
            }

            // Extract features based on method
            vector<float> features;
            if(method == "baseline") {
                features = extractBaselineFeatures(img);
            } else if(method == "histogram") {
                features = extractColorHistogramFeatures(img);
            } else if(method == "multihist") {
                features = extractMultiHistogramFeatures(img);
            } else if(method == "texture") {
                features = extractTextureColorFeatures(img);
            }
            // Compute distance based on method
            float distance;
            if(method == "baseline") {
                distance = computeSSD(targetFeatures, features);
            } else if(method == "histogram") {
                distance = histogramIntersection(targetFeatures, features);
            } else if(method == "multihist") {
                distance = multiHistogramDistance(targetFeatures, features);
            } else if(method == "texture") {
                distance = textureColorDistance(targetFeatures, features);
             }
            matches.push_back(ImageMatch(dp->d_name, distance));
            imageCount++;
        }
    }

    closedir(dirp);
    cout << "Processed " << imageCount << " images" << endl;

    // Sort and display results
    sort(matches.begin(), matches.end(),
         [](const ImageMatch &a, const ImageMatch &b) {
             return a.distance < b.distance;
         });

    cout << "Top " << N << " matches:" << endl;
    for(int i = 0; i < N && i < static_cast<int>(matches.size()); i++) {
        cout << (i+1) << ". " << matches[i].filename
             << " (distance: " << matches[i].distance << ")" << endl;
    }

    return 0;
}