import os
import sys
import torch
import cv2
import numpy as np
import logging
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ultralytics import YOLO
from scipy import stats

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('fuzzy_elephant_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EnhancedFuzzyConfidenceController:
    """
    Advanced Fuzzy Logic Confidence Controller for Intelligent Object Detection

    This class implements a sophisticated fuzzy inference system that adaptively
    modifies detection confidence based on multiple contextual parameters.
    """

    def __init__(self, base_confidence=0.5):
        """
        Initialize the fuzzy confidence controller with adaptive parameters.

        Args:
            base_confidence (float): Initial confidence threshold
        """
        self.base_confidence = base_confidence
        self.previous_detections = []
        self._create_enhanced_fuzzy_system()

    def _calculate_object_complexity(self, frame, bbox):
        """
        Calculate local image complexity around detected object.

        Uses entropy as a measure of local region complexity, providing
        a nuanced understanding of the detection environment.

        Args:
            frame (numpy.ndarray): Input video frame
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)

        Returns:
            float: Normalized complexity metric
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            # Convert to grayscale for entropy calculation
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Calculate entropy to measure local complexity
            entropy = stats.entropy(np.histogram(gray_roi, bins=256)[0])
            return min(1.0, entropy / 8.0)  # Normalize
        return 0.5

    def _calculate_object_relative_size(self, frame, bbox):
        """
        Compute object size relative to entire frame.

        Provides a normalized representation of object size,
        crucial for intelligent confidence adjustment.

        Args:
            frame (numpy.ndarray): Input video frame
            bbox (tuple): Bounding box coordinates

        Returns:
            float: Normalized object size
        """
        frame_area = frame.shape[0] * frame.shape[1]
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return min(1.0, bbox_area / frame_area)

    def _create_enhanced_fuzzy_system(self):
        """
        Create a comprehensive fuzzy inference system with
        nuanced rules and membership functions.
        """
        try:
            # Define universe of discourse
            universe = np.linspace(0, 1, 100)

            # Input Variables with Advanced Categorization
            self.object_complexity = ctrl.Antecedent(universe, 'object_complexity')
            self.object_size = ctrl.Antecedent(universe, 'object_size')
            self.detection_history = ctrl.Antecedent(universe, 'detection_history')

            # Output: Confidence Adjustment
            self.confidence_adjustment = ctrl.Consequent(
                np.linspace(-0.3, 0.3, 100),
                'confidence_adjustment'
            )

            # MODIFICATION: Define membership functions directly on the Antecedent

            # Detailed Membership Functions
            self.object_complexity['very_simple'] = fuzz.trimf(self.object_complexity.universe, [0, 0, 0.2])
            self.object_complexity['simple'] = fuzz.trimf(self.object_complexity.universe, [0.1, 0.3, 0.5])
            self.object_complexity['moderate'] = fuzz.trimf(self.object_complexity.universe, [0.4, 0.6, 0.8])
            self.object_complexity['complex'] = fuzz.trimf(self.object_complexity.universe, [0.7, 0.9, 1])

            self.object_size['very_small'] = fuzz.trimf(self.object_size.universe, [0, 0, 0.2])
            self.object_size['small'] = fuzz.trimf(self.object_size.universe, [0.1, 0.3, 0.5])
            self.object_size['medium'] = fuzz.trimf(self.object_size.universe, [0.4, 0.6, 0.8])
            self.object_size['large'] = fuzz.trimf(self.object_size.universe, [0.7, 0.9, 1])

            self.detection_history['unstable'] = fuzz.trimf(self.detection_history.universe, [0, 0, 0.3])
            self.detection_history['moderate'] = fuzz.trimf(self.detection_history.universe, [0.2, 0.5, 0.8])
            self.detection_history['consistent'] = fuzz.trimf(self.detection_history.universe, [0.7, 1, 1])

            self.confidence_adjustment['strong_reduce'] = fuzz.trimf(self.confidence_adjustment.universe, [-0.3, -0.2, -0.1])
            self.confidence_adjustment['reduce'] = fuzz.trimf(self.confidence_adjustment.universe, [-0.2, -0.1, 0])
            self.confidence_adjustment['maintain'] = fuzz.trimf(self.confidence_adjustment.universe, [-0.1, 0, 0.1])
            self.confidence_adjustment['increase'] = fuzz.trimf(self.confidence_adjustment.universe, [0, 0.1, 0.2])
            self.confidence_adjustment['strong_increase'] = fuzz.trimf(self.confidence_adjustment.universe, [0.1, 0.2, 0.3])

            # Comprehensive Fuzzy Rules
            rules = [
                # Complex scenes with small objects: Reduce confidence
                ctrl.Rule(
                    self.object_complexity['complex'] &
                    self.object_size['very_small'],
                    self.confidence_adjustment['strong_reduce']
                ),

                # Large objects in simple scenes: Increase confidence
                ctrl.Rule(
                    self.object_complexity['very_simple'] &
                    self.object_size['large'],
                    self.confidence_adjustment['strong_increase']
                ),

                # Unstable detection history reduces confidence
                ctrl.Rule(
                    self.detection_history['unstable'],
                    self.confidence_adjustment['reduce']
                ),

                # Consistent detections in moderate complexity
                ctrl.Rule(
                    self.object_complexity['moderate'] &
                    self.detection_history['consistent'],
                    self.confidence_adjustment['maintain']
                )
            ]

            # Create Control System
            self.system = ctrl.ControlSystem(rules)
            self.simulator = ctrl.ControlSystemSimulation(self.system)

        except Exception as e:
            logger.error(f"Enhanced Fuzzy System Initialization Failed: {e}")
            raise

    def update_detection_history(self, detection_confidences):
        """
        Update and manage detection history for intelligent tracking.

        Args:
            detection_confidences (list): Recent detection confidences
        """
        self.previous_detections.extend(detection_confidences)
        self.previous_detections = self.previous_detections[-10:]

    def compute_detection_history_stability(self):
        """
        Compute the stability of detection history.

        Returns:
            float: Normalized detection history stability
        """
        if not self.previous_detections:
            return 0.5  # Neutral default value

        stability = 1 - np.std(self.previous_detections)
        return max(0, min(1, stability))

    def adjust_confidence(
            self,
            frame,
            bbox,
            current_confidence,
            detection_confidences=None
    ):
        """
        Dynamically adjust detection confidence using enhanced fuzzy logic.

        Args:
            frame (numpy.ndarray): Current video frame
            bbox (tuple): Detected object bounding box
            current_confidence (float): Current detection confidence
            detection_confidences (list, optional): Recent detection confidences

        Returns:
            float: Adaptively adjusted confidence threshold
        """
        try:
            # Update detection history
            if detection_confidences:
                self.update_detection_history(detection_confidences)

            # Compute contextual parameters
            object_complexity = self._calculate_object_complexity(frame, bbox)
            object_size = self._calculate_object_relative_size(frame, bbox)
            detection_history_stability = self.compute_detection_history_stability()

            # Set fuzzy input values
            self.simulator.input['object_complexity'] = object_complexity
            self.simulator.input['object_size'] = object_size
            self.simulator.input['detection_history'] = detection_history_stability

            # Compute fuzzy inference
            self.simulator.compute()

            # Retrieve confidence adjustment
            adjustment = self.simulator.output['confidence_adjustment']

            # Apply adjustment to current confidence
            adjusted_conf = max(0.01, min(0.99, current_confidence + adjustment))

            return adjusted_conf

        except Exception as e:
            logger.warning(f"Enhanced Fuzzy Confidence Adjustment Failed: {e}")
            return self.base_confidence


class FuzzyElephantDetector:
    """
    Intelligent Elephant Detection System with Advanced Fuzzy Logic Enhancement
    """

    def __init__(
            self,
            model_path=None,
            base_confidence=0.5,
            device=None
    ):
        """
        Initialize the advanced fuzzy elephant detector.

        Args:
            model_path (str): Path to custom YOLO model
            base_confidence (float): Base detection confidence
            device (str): Computation device
        """
        # Device selection with intelligent fallback
        if device is None:
            device = (
                'cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else
                'cpu'
            )

        # Model and fuzzy controller initialization
        try:
            self.model = YOLO(model_path or 'yolov8n.pt')
            self.device = device

            # Advanced Fuzzy Confidence Controller
            self.fuzzy_controller = EnhancedFuzzyConfidenceController(
                base_confidence=base_confidence
            )

            logger.info(f"Fuzzy Elephant Detector initialized on {device}")

        except Exception as e:
            logger.error(f"Detector Initialization Failed: {e}")
            raise

    def detect_elephants_in_video(
            self,
            video_path,
            output_path='fuzzy_elephant_detection.mp4',
            save_crops=True
    ):
        """
        Detect and track elephants in video with intelligent fuzzy logic.

        Args:
            video_path (str): Input video file path
            output_path (str): Output annotated video path
            save_crops (bool): Save individual elephant detection crops
        """
        # Video processing setup
        cap = cv2.VideoCapture(video_path)

        # Video writer configuration
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        # Tracking variables
        frame_count = 0
        elephant_count = 0
        detection_confidences = []

        # Crops directory
        if save_crops:
            os.makedirs('fuzzy_elephant_crops', exist_ok=True)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Run initial YOLO detection
                results = self.model(frame, conf=0.3, verbose=False)

                # Process detections
                if results and len(results[0].boxes) > 0:
                    for i, box in enumerate(results[0].boxes):
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]

                        # Adjust confidence using fuzzy logic
                        adjusted_conf = self.fuzzy_controller.adjust_confidence(
                            frame,
                            (x1, y1, x2, y2),
                            conf,
                            detection_confidences
                        )

                        # Store detection confidence
                        detection_confidences.append(adjusted_conf)

                        # Elephant detection (adjust class index as needed)
                        if cls == 0 and adjusted_conf > 0.4:
                            elephant_count += 1

                            # Adaptive bounding box visualization
                            box_color = (0, int(255 * adjusted_conf), 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                            # Confidence annotation
                            label = f'Elephant {adjusted_conf:.2f}'
                            cv2.putText(
                                frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2
                            )

                            # Save elephant crops
                            if save_crops:
                                crop = frame[y1:y2, x1:x2]
                                crop_path = f'fuzzy_elephant_crops/elephant_{frame_count}_{i}.jpg'
                                cv2.imwrite(crop_path, crop)

                # Write annotated frame
                out.write(frame)

                # Progress logging
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames")

        except Exception as e:
            logger.error(f"Detection Process Error: {e}")

        finally:
            # Resource cleanup
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        # Final detection report
        logger.info("Fuzzy Elephant Detection Completed")
        logger.info(f"Total Frames Processed: {frame_count}")
        logger.info(f"Total Elephants Detected: {elephant_count}")


def main():
    """
    Main execution function for advanced fuzzy elephant detection
    """
    try:
        # Initialize enhanced fuzzy elephant detector
        detector = FuzzyElephantDetector(
            model_path='/Users/ravindupabasarakarunarathna/Documents/Self_Studies/fuzzy/Fuzzy/elephant_detection3/weights/best.pt',
            base_confidence=0.3
        )

        # Detect elephants in video
        detector.detect_elephants_in_video(
            video_path='test_vid4.mp4',
            output_path='enhanced_fuzzy_elephant_detection.mp4'
        )

    except Exception as e:
        logger.error(f"Execution Failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()