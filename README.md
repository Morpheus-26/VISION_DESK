# VISION_DESK
This Python application provides real-time video processing with the ability to toggle between the original camera feed and Canny Edge Detection. It allows users to select a Region of Interest (ROI) and interactively tune threshold values for edge detection.

# How the Code Works

• **Image Acquisition**:
  - Captures frames from webcam using `cv2.VideoCapture(0)`
  - Main loop continuously reads frames with `cap.read()`

• **Processing Pipeline**:
  - Each frame passes through the selected filter function
  - Functions like `apply_canny()`, `apply_sepia()` transform the pixel data
  - Results are stored in `display_frame`.

• **ROI Handling**:
  - Mouse events captured through `cv2.setMouseCallback`
  - ROI coordinates stored in `self.roi_rect` as (x, y, width, height)

• **Filter Implementation**:
  - Edge detection: Converts to grayscale → Gaussian blur → Canny algorithm
  - Grayscale: Simple color conversion with `cv2.cvtColor`
  - Sepia: Matrix multiplication with color transformation values
  - Blur: Applies Gaussian blur with kernel size (15,15)
  - Face detection: Uses Haar cascade classifier to find face regions

• **User Input Processing**:
  - Keyboard events captured with `cv2.waitKey(1)`
  - Number keys change `self.mode` variable
  - Mode determines which filter function is called

• **Performance Tracking**:
  - Calculates time difference between frames
  - Converts to FPS using 1/time_difference formula

• **Display Output**:
  - Adds text overlays with `cv2.putText`
  - Shows processed frames via `cv2.imshow`
