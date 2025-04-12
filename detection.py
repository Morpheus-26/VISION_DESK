import cv2
import numpy as np
import time

class EnhancedEdgeDetectionApp:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            exit()
        
        # Default parameters
        self.mode = "original"  # "original", "edge", "grayscale", "sepia", "blur", "face_detect"
        self.roi_selecting = False
        self.roi_selected = False
        self.roi_points = []
        self.roi_rect = None
        self.lower_threshold = 100
        self.upper_threshold = 200
        
        # FPS calculation
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create window and trackbars
        cv2.namedWindow('Enhanced Edge Detection App')
        cv2.createTrackbar('Lower Threshold', 'Enhanced Edge Detection App', self.lower_threshold, 255, self.on_lower_threshold_change)
        cv2.createTrackbar('Upper Threshold', 'Enhanced Edge Detection App', self.upper_threshold, 255, self.on_upper_threshold_change)
        
        # Mouse callback
        cv2.setMouseCallback('Enhanced Edge Detection App', self.mouse_callback)
        
        print("Controls:")
        print("  '1' - Original mode")
        print("  '2' - Edge detection mode")
        print("  '3' - Grayscale mode")
        print("  '4' - Sepia mode")
        print("  '5' - Blur mode")
        print("  '6' - Face detection mode")
        print("  'r' - Start/reset ROI selection")
        print("  'c' - Clear ROI")
        print("  'q' - Quit application")
        print("  Use trackbars to adjust thresholds for edge detection")
        print("  Click and drag to select ROI")
    
    def on_lower_threshold_change(self, value):
        self.lower_threshold = value
    
    def on_upper_threshold_change(self, value):
        self.upper_threshold = value
    
    def mouse_callback(self, event, x, y, flags, param):
        if self.roi_selecting:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_points = [(x, y)]
            elif event == cv2.EVENT_LBUTTONUP:
                self.roi_points.append((x, y))
                x1, y1 = self.roi_points[0]
                x2, y2 = self.roi_points[1]
                self.roi_rect = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                self.roi_selecting = False
                self.roi_selected = True
    
    def apply_canny(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.lower_threshold, self.upper_threshold)
        # Convert back to BGR for display
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def apply_grayscale(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def apply_sepia(self, frame):
        # Convert to float and normalize
        frame_normalized = np.array(frame, dtype=np.float32) / 255.0
        
        # Sepia filter matrix
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Apply the sepia matrix
        sepia_img = cv2.transform(frame_normalized, sepia_matrix)
        
        # Clip values to stay within 0-1 range
        sepia_img = np.clip(sepia_img, 0, 1)
        
        # Convert back to uint8
        sepia_img = np.array(sepia_img * 255, dtype=np.uint8)
        
        return sepia_img
    
    def apply_blur(self, frame):
        return cv2.GaussianBlur(frame, (15, 15), 0)
    
    def apply_face_detection(self, frame):
        # Create a copy for drawing
        result = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return result
    
    def process_frame(self, frame, mode):
        if mode == "edge":
            return self.apply_canny(frame)
        elif mode == "grayscale":
            return self.apply_grayscale(frame)
        elif mode == "sepia":
            return self.apply_sepia(frame)
        elif mode == "blur":
            return self.apply_blur(frame)
        elif mode == "face_detect":
            return self.apply_face_detection(frame)
        else:  # original
            return frame
    
    def calculate_fps(self):
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if (self.curr_frame_time - self.prev_frame_time) > 0 else 0
        self.prev_frame_time = self.curr_frame_time
        return round(fps, 1)
    
    def run(self):
        while True:
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Calculate FPS
            self.fps = self.calculate_fps()
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Process ROI if selected
            if self.roi_selected:
                x, y, w, h = self.roi_rect
                roi = frame[y:y+h, x:x+w]
                
                if self.mode != "original":
                    roi_processed = self.process_frame(roi, self.mode)
                    display_frame[y:y+h, x:x+w] = roi_processed
                
                # Draw rectangle around ROI
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Process entire frame if no ROI is selected
            elif self.mode != "original":
                display_frame = self.process_frame(frame, self.mode)
            
            # Show current mode
            mode_text = f"Mode: {self.mode.replace('_', ' ').title()}"
            cv2.putText(display_frame, mode_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show FPS
            fps_text = f"FPS: {self.fps}"
            cv2.putText(display_frame, fps_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display ROI selection instructions if selecting
            if self.roi_selecting:
                cv2.putText(display_frame, "Click and drag to select ROI", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Enhanced Edge Detection App', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.mode = "original"
            elif key == ord('2'):
                self.mode = "edge"
            elif key == ord('3'):
                self.mode = "grayscale"
            elif key == ord('4'):
                self.mode = "sepia"
            elif key == ord('5'):
                self.mode = "blur"
            elif key == ord('6'):
                self.mode = "face_detect"
            elif key == ord('r'):
                self.roi_selecting = True
                self.roi_selected = False
            elif key == ord('c'):
                self.roi_selected = False
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EnhancedEdgeDetectionApp()
    app.run()