import cv2
import numpy as np
from LaneAnalyzer import LaneAnalyzer
import time
import os
import tkinter as tk # Import tkinter
from tkinter import ttk

# --- MODEL CONFIGURATION ---
MODEL_OPTIONS = {
    "YOLOv10 Nano (Fastest)": "yolov10n.pt",
    "YOLOv10 Small (Balanced)": "yolov10s.pt",
    "YOLOv10 Medium (Accurate)": "yolov10m.pt",
    "YOLOv10 Large (Most Accurate)": "yolov10l.pt"
}
# Default model path if selection fails
DEFAULT_MODEL_PATH = "yolov10n.pt" 
# --- END MODEL CONFIGURATION ---

# MODEL_PATH will be set by the Tkinter window
VIDEO_PATH = "test.mp4"
ROI_WINDOW_NAME = "Select Congestion Zone"

MAX_DISAPPEARED = 60
MAX_DISTANCE = 80

roi_points = []
display_frame = None
original_frame = None

def get_model_choice():
    """Opens a Tkinter window to let the user select the YOLO model."""
    
    # Store the user's choice
    chosen_model_file = [DEFAULT_MODEL_PATH]

    def select_and_close():
        # Get the friendly name from the combobox
        selected_key = model_combo.get()
        # Look up the corresponding file path
        chosen_model_file[0] = MODEL_OPTIONS.get(selected_key, DEFAULT_MODEL_PATH)
        root.destroy()

    root = tk.Tk()
    root.title("YOLO Model Selector")
    
    # Center the window (optional, but nice)
    window_width = 350
    window_height = 120
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # Label
    label = ttk.Label(root, text="Choose a YOLOv10 Model:")
    label.pack(pady=10)

    # Combobox (Dropdown)
    model_combo = ttk.Combobox(root, values=list(MODEL_OPTIONS.keys()), state="readonly", width=35)
    model_combo.set(list(MODEL_OPTIONS.keys())[0]) # Set default selection
    model_combo.pack(pady=5)

    # Button
    select_button = ttk.Button(root, text="Start Analysis", command=select_and_close)
    select_button.pack(pady=10)

    root.mainloop()
    
    return chosen_model_file[0]


def draw_roi(frame, points):
    if not points:
        return
    
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
    
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], (0, 255, 0), 2)
            
    if len(points) == 4:
        cv2.line(frame, points[3], points[0], (0, 255, 0), 2)

def mouse_callback(event, x, y, flags, param):
    global roi_points, display_frame, original_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(roi_points) < 4:
            roi_points.append((x, y))
            print(f"Point {len(roi_points)} selected: {(x, y)}")
            
            display_frame = original_frame.copy()
            draw_roi(display_frame, roi_points)
            
            if len(roi_points) == 4:
                print("4 points selected. Press 'c' to confirm, or 'r' to reset.")
        else:
            print("Already 4 points selected. Press 'r' to reset.")

def select_roi(first_frame):
    global roi_points, display_frame, original_frame
    
    if first_frame is None:
        print("ERROR in select_roi: first_frame is None.")
        return None
        
    roi_points = []
    original_frame = first_frame.copy()
    display_frame = first_frame.copy()
    
    # --- Window Creation (Optimized for drawing) ---
    cv2.namedWindow(ROI_WINDOW_NAME, cv2.WINDOW_NORMAL) 
    cv2.setWindowProperty(ROI_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
    
    cv2.setMouseCallback(ROI_WINDOW_NAME, mouse_callback) 
    
    cv2.imshow(ROI_WINDOW_NAME, display_frame) 
    cv2.waitKey(1) 

    print("Please select 4 points for the congestion zone.")
    print("Click to select a point.")
    print("Press 'r' to reset points.")
    print("Press 'c' to confirm (after 4 points).")
    print("Press 'q' to quit.")

    while True:
        cv2.imshow(ROI_WINDOW_NAME, display_frame) 
        key = cv2.waitKey(33) & 0xFF 

        if key == ord('q'):
            cv2.destroyWindow(ROI_WINDOW_NAME)
            return None
        elif key == ord('r'):
            roi_points = []
            display_frame = original_frame.copy()
            print("Points reset. Please select 4 new points.")
        elif key == ord('c'):
            if len(roi_points) == 4:
                print("ROI Confirmed.")
                break
            else:
                print("Please select 4 points before confirming.")

    cv2.destroyWindow(ROI_WINDOW_NAME)
    return roi_points

def main():
    
    # 1. --- MODEL SELECTION (NEW) ---
    chosen_model_path = get_model_choice()
    MODEL_PATH = chosen_model_path
    print(f"Using model: {MODEL_PATH}")
    # ---------------------------------
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    is_image = False
    if isinstance(VIDEO_PATH, str) and not VIDEO_PATH.startswith('http') and not VIDEO_PATH.startswith('rtsp'):
        if os.path.splitext(VIDEO_PATH)[1].lower() in image_extensions:
            is_image = True

    if is_image:
        print(f"Processing single image file: {VIDEO_PATH}")
        first_frame = cv2.imread(VIDEO_PATH)
        if first_frame is None:
            print(f"Error: Could not read image file {VIDEO_PATH}")
            return
        video = None
        ret = True
        
    else:
        print(f"Connecting to video stream: {VIDEO_PATH}")
        video = cv2.VideoCapture(VIDEO_PATH)
        
        if not video.isOpened():
            print(f"Error: Could not open video {VIDEO_PATH}")
            return

        print("Successfully connected. Grabbing first frame...")
        ret, first_frame = video.read()
        if not ret:
            print("Error: Could not read first frame from stream.")
            video.release()
            return
    
    MAIN_WINDOW_NAME = "Traffic Congestion Analyzer"

    print("Frame grabbed. Please select ROI.")

    user_roi_points = select_roi(first_frame)

    if user_roi_points is None:
        print("Setup cancelled. Exiting.")
        if video: video.release()
        cv2.destroyAllWindows()
        return

    # Window Creation (Main Analysis Window)
    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(MAIN_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    analyzer = LaneAnalyzer(model_path=MODEL_PATH, roi_points=user_roi_points, max_disappeared=MAX_DISAPPEARED, max_distance=MAX_DISTANCE, conf_threshold=0.2)
    start_time = time.time()
    frame_count = 0
    print("Starting traffic analysis...")

    while True:
        
        if frame_count == 0:
            current_frame = first_frame
            ret = True
        elif is_image:
            ret = False
            current_frame = None
        else:
            ret, current_frame = video.read()
            
        if not ret:
            break

        score, annotated_frame = analyzer.process_frame(current_frame)
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (5, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(MAIN_WINDOW_NAME, annotated_frame)

        if is_image:
            print("Analysis complete. Press 'q' or close window to exit.")
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    if video: video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
