
"""
Streamlit GUI
Astitva Goel
Took help from Claude to connect C++ backend
Web interface for the CBIR system. Communicates with C++ backend
via subprocess calls and displays results in a modern web interface.
"""

import streamlit as st
from PIL import Image
import subprocess
import os
from pathlib import Path
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="CBIR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# System configuration
EXECUTABLE = "../build/Project2/Project2"  # Path to C++ executable
DATABASE = "olympus"                       # Image database directory
CSV_FILE = "ResNet18_olym.csv"            # ResNet18 embeddings file

def parse_output(output):
    """
    Parse C++ program stdout to extract match results
    
    Expected format:
        Top 5 matches:
        ========================================
        1. pic.0893.jpg (distance: 0.123)
        2. pic.0456.jpg (distance: 0.234)
        ...
    
    Returns: List of (filename, distance) tuples
    """
    matches = []
    in_results = False
    
    for line in output.split('\n'):
        # Detect start of results section
        if "matches:" in line.lower() or "========" in line:
            in_results = True
            continue
        
        if in_results and line.strip():
            # Parse line like "1. pic.0893.jpg (distance: 0.123)"
            if '. ' in line and '(' in line:
                try:
                    parts = line.split('. ', 1)[1].split(' (distance: ')
                    filename = parts[0].strip()
                    distance = parts[1].rstrip(')')
                    matches.append((filename, float(distance)))
                except:
                    continue
    
    return matches

def run_search(target_path, method, num_results):
    """
    Execute C++ CBIR backend as subprocess
    
    Args:
        target_path: Path to target image
        method: Matching method (baseline, histogram, etc.)
        num_results: Number of matches to return
    
    Returns:
        Tuple of (matches_list, raw_output) or (None, error_message)
    """
    try:
        # Build command line arguments
        cmd = [
            EXECUTABLE,
            target_path,
            DATABASE,
            method,
            str(num_results)
        ]
        
        # Add CSV file for methods that need DNN embeddings
        if method in ["embedding", "adaptive", "trashcan", "custom"]:
            cmd.append(CSV_FILE)
        
        # Run C++ program with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return parse_output(result.stdout), result.stdout
        else:
            return None, result.stderr
            
    except subprocess.TimeoutExpired:
        return None, "Search timed out (>120 seconds)"
    except Exception as e:
        return None, str(e) 

def main():
    # ==================== HEADER ====================
    col1, col2, col3 = st.columns([1, 9, 1])
    with col2:
        st.header("Content-Based Image Retrieval System")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ==================== SIDEBAR - CONTROLS ====================
    with st.sidebar:
        st.markdown("## Search Settings")
        
        # Method selection dropdown
        method_options = {
            "Deep Learning (DNN)": "embedding",
            "Baseline (7×7 pixels)": "baseline",
            "Color Histogram": "histogram",
            "Multi-Histogram": "multihist",
            "Texture + Color": "texture",
            "Adaptive (Face-aware)": "adaptive",
            "Trash Can Finder": "trashcan",
            "Custom Nature Scenes": "custom"
        }
        
        selected_method_name = st.selectbox(
            "Matching Method",
            options=list(method_options.keys()),
            index=0,
            help="Choose the algorithm for finding similar images"
        )
        method = method_options[selected_method_name]
        
        st.markdown("")
        
        # Number of results slider
        num_results = st.slider(
            "Number of Results",
            min_value=1,
            max_value=50,
            value=5,
            help="How many similar images to find"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Method information box
        st.markdown("### Method Info")
        method_descriptions = {
            "embedding": "Uses ResNet18 deep learning features trained on ImageNet. Best for semantic similarity across different object categories.",
            "baseline": "Compares 7×7 center pixel values. Fast but limited to exact matches.",
            "histogram": "Matches overall color distribution. Good for images with similar color schemes.",
            "multihist": "Compares top and bottom color regions. Captures spatial layout like sky/ground.",
            "texture": "Combines gradient patterns with color histograms. Good for textured surfaces.",
            "adaptive": "Detects faces and prioritizes them. Ideal for finding people or portraits.",
            "trashcan": "Specialized detector for blue trash bins using color and shape features.",
            "custom": "Optimized for outdoor nature scenes with sky, vegetation, and ground regions."
        }
        st.info(method_descriptions[method])
        
        st.markdown("<hr>", unsafe_allow_html=True)

    # ==================== MAIN CONTENT ====================
    col1, col2 = st.columns([1, 2], gap="large")
    
    # LEFT COLUMN - Target Selection
    with col1:
        st.markdown("## Target Image")
        
        # Get list of images from database
        if os.path.exists(DATABASE):
            image_files = sorted([f for f in os.listdir(DATABASE)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.tif'))])

            selected_file = st.selectbox(
                "Select from database",
                options=[""] + image_files,
                format_func=lambda x: "Choose an image..." if x == "" else x,
                help="Pick an image from the olympus database"
            )
        else:
            st.error(f"Database directory '{DATABASE}' not found!")
            return

        # Display selected target image
        target_path = None
        if selected_file != "":
            target_path = os.path.join(DATABASE, selected_file)
            image = Image.open(target_path)
            st.image(image, use_column_width=True, caption=selected_file)

        st.markdown("")

        # Search button
        if target_path:
            search_clicked = st.button("Search for Similar Images", type="primary", use_container_width=True)
            
            if search_clicked:
                with st.spinner('Searching database...'):
                    matches, output = run_search(target_path, method, num_results)
                    
                    if matches:
                        # Store results in session state
                        st.session_state.matches = matches
                        st.session_state.search_output = output
                        st.success(f"Found {len(matches)} matches!")
                    else:
                        st.error(f"Search failed: {output}")
        else:
            st.info("Select an image to begin searching")
    
    # RIGHT COLUMN - Results Display
    with col2:
        st.markdown("## Search Results")
        
        # Display results if available in session state
        if 'matches' in st.session_state and st.session_state.matches:
            matches = st.session_state.matches
            
            # Performance metrics
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Total Results", len(matches))
            with metric_cols[1]:
                st.metric("Best Match", f"{matches[0][1]:.4f}")
            with metric_cols[2]:
                avg_dist = sum(d for _, d in matches) / len(matches)
                st.metric("Avg Distance", f"{avg_dist:.4f}")
            
            st.markdown("")
            
            # ========== TOP MATCHES (Most Similar) ===========
            st.markdown("### Most Similar Images")

            # Display top 5 in grid layout
            top_matches = matches[:min(5, len(matches))]

            for i in range(0, len(top_matches), 3):
                cols = st.columns(3, gap="medium")

                for j, col in enumerate(cols):
                    if i + j < len(top_matches):
                        filename, distance = top_matches[i + j]
                        rank = i + j + 1

                        with col:
                            try:
                                img_path = os.path.join(DATABASE, filename)
                                img = Image.open(img_path)
                                st.image(img, use_column_width=True)

                                # Green card for similar matches
                                st.markdown(f"""
                                <div style='
                                    text-align: center; 
                                    padding: 10px; 
                                    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                                    border-radius: 8px;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    margin-top: -10px;
                                    border: 2px solid #28a745;
                                '>
                                    <div style='font-size: 1.2em; font-weight: bold; color: #155724;'>
                                        #{rank}
                                    </div>
                                    <div style='font-size: 0.9em; color: #155724; margin: 5px 0;'>
                                        {filename}
                                    </div>
                                    <div style='font-size: 0.85em; color: #28a745; font-weight: 600;'>
                                        Distance: {distance:.4f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            except Exception as e:
                                st.error(f"Error loading {filename}")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ========== LEAST SIMILAR MATCHES ==========
            st.markdown("### Least Similar Images")

            # Show bottom 5 if we have enough results
            if len(matches) > 5:
                least_matches = matches[-5:]
                
                for i in range(0, len(least_matches), 3):
                    cols = st.columns(3, gap="medium")
                    
                    for j, col in enumerate(cols):
                        if i + j < len(least_matches):
                            filename, distance = least_matches[i + j]
                            rank = len(matches) - len(least_matches) + i + j + 1
                            
                            with col:
                                try:
                                    img_path = os.path.join(DATABASE, filename)
                                    img = Image.open(img_path)
                                    st.image(img, use_column_width=True)
                                    
                                    # Red card for dissimilar matches
                                    st.markdown(f"""
                                    <div style='
                                        text-align: center; 
                                        padding: 10px; 
                                        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                                        border-radius: 8px;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                        margin-top: -10px;
                                        border: 2px solid #dc3545;
                                    '>
                                        <div style='font-size: 1.2em; font-weight: bold; color: #721c24;'>
                                            #{rank}
                                        </div>
                                        <div style='font-size: 0.9em; color: #721c24; margin: 5px 0;'>
                                            {filename}
                                        </div>
                                        <div style='font-size: 0.85em; color: #dc3545; font-weight: 600;'>
                                            Distance: {distance:.4f}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                except Exception as e:
                                    st.error(f"Error loading {filename}")
            else:
                st.info("Need more than 5 results to show least similar matches")
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # ========== ALL RESULTS (Expandable) ==========
            if len(matches) > 10:
                with st.expander(f"View All {len(matches)} Results"):
                    for i, (filename, distance) in enumerate(matches):
                        col1, col2, col3 = st.columns([1, 3, 2])
                        
                        # Thumbnail
                        with col1:
                            try:
                                img_path = os.path.join(DATABASE, filename)
                                img = Image.open(img_path)
                                st.image(img, width=100)
                            except:
                                st.write("Error")
                        
                        # Filename
                        with col2:
                            st.write(f"**{filename}**")
                        
                        # Distance
                        with col3:
                            st.write(f"Distance: {distance:.4f}")
            
            # Raw program output for debugging
            with st.expander("View Raw Output"):
                st.code(st.session_state.search_output, language="text")
                
        else:
            # Empty state message
            st.info("""
            **Get Started:**
            1. Select a target image from the sidebar
            2. Choose a matching method
            3. Click the Search button
            4. View your results here!
            """)
    
   

if __name__ == "__main__":
    main()