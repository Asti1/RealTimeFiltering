## Student Information

**Name:** Astitva Goel  
**Group Members:** None (Individual Project)  
**Date:** February 9, 2025

## Development Environment

**Operating System:** macOS (Sonoma/Ventura)  
**IDE:** Visual Studio Code  
**Compiler:** clang++ (Apple clang version XX.X)  
**Build System:** CMake 3.10+  
**Libraries:** OpenCV 4.13.0

## Building the Project

### Prerequisites

```bash
# Install OpenCV via Homebrew
brew install opencv cmake

# Install Python dependencies for GUI (optional)
pip3 install streamlit pillow
```

### Compilation Steps

```bash
# Navigate to build directory
cd ~/PRCV/build/Project2

# Clean previous build (if needed)
rm -rf *

# Configure and build
cmake ../../Project2
make
```

## Building the Project

### Prerequisites

```bash
# Install OpenCV via Homebrew
brew install opencv cmake

# Install Python dependencies for GUI (optional)
pip3 install streamlit pillow
```

### Compilation Steps

```bash
# Navigate to build directory
cd ~/PRCV/build/Project2

# Clean previous build (if needed)
rm -rf *

# Configure and build
cmake ../../Project2
make
```

### Example Commands

**From build/Project2 directory:**

```bash
# Task 1: Baseline matching
./Project2 ../../Project2/olympus/pic.1016.jpg ../../Project2/olympus baseline 5

# Task 2: Histogram matching
./Project2 ../../Project2/olympus/pic.XXXX.jpg ../../Project2/olympus histogram 5

# Task 3: Multi-histogram matching
./Project2 ../../Project2/olympus/pic.0274.jpg ../../Project2/olympus multihist 5

# Task 4: Texture and color
./Project2 ../../Project2/olympus/pic.0535.jpg ../../Project2/olympus texture 5

# Task 5: DNN embeddings
./Project2 ../../Project2/olympus/pic.0893.jpg ../../Project2/olympus embedding 10 ../../Project2/ResNet18_olym.csv

```

**Running the GUI:**

```bash
# Navigate to Project2 directory
cd ~/PRCV/Project2

# Launch Streamlit
streamlit run cbir_app.py

# Browser will open automatically to http://localhost:8501
```
