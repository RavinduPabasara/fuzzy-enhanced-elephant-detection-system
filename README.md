# **Fuzzy Elephant Detection System**  

## **Overview**  
This project implements an advanced object detection system designed to locate wild elephants in rural areas, aiming to reduce Human-Elephant Conflict (HEC). By combining state-of-the-art YOLO object detection with a fuzzy logic-based confidence controller, the system adapts to environmental complexities and improves detection reliability.  

## **Features**  
- Adaptive confidence adjustment using fuzzy logic to handle diverse detection scenarios.  
- Intelligent detection of elephants in real-time from video streams.  
- Context-aware decision-making based on object size, complexity, and detection stability.  
- Saves cropped images of detected elephants for further analysis.  

## **Technologies Used**  
- **YOLO**: Efficient object detection model.  
- **Python**: Primary programming language.  
- **OpenCV**: For video processing and visualization.  
- **Skfuzzy**: Fuzzy logic control for confidence adjustment.  
- **NumPy** and **SciPy**: For numerical and statistical computations.  
- **Torch**: Deep learning framework for model inference.  

## **How It Works**  
1. Detects elephants in video frames using a YOLO-based object detection model.  
2. Dynamically adjusts detection confidence using fuzzy logic based on:  
   - Object complexity (calculated using image entropy).  
   - Object size relative to the frame.  
   - Stability of detection history.  
3. Saves annotated video output and optional cropped elephant images.  

## **Installation**  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/fuzzy-elephant-detection.git  
   cd fuzzy-elephant-detection  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Place your YOLO model weights file in the appropriate directory.  

## **Usage**  
To run the system on a video file:  
```bash  
python main.py --video_path <path_to_video> --output_path <output_video_path>  
```  
Example:  
```bash  
python main.py --video_path test_video.mp4 --output_path output.mp4  
```  

## **Applications**  
- Early detection and alert systems to prevent Human-Elephant Conflict (HEC).  
- Wildlife conservation and monitoring in rural areas.  

## **Contributing**  
Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.  

## **License**  
This project is licensed under the MIT License.  

