import cv2
import numpy as np
from LaneAnalyzer import LaneAnalyzer
import time
import os
import tkinter as tk
from tkinter import ttk, filedialog # Import filedialog

MODEL_OPTIONS = {
    "YOLO11 Semantic Segmentation": "yolo11n-seg.pt"
}
DEFAULT_MODEL_PATH = "yolo11n-seg.pt" 

# VIDEO_PATH = "test.mp4" 
ROI_WINDOW_NAME = "Select Congestion Zone"

MAX_DISAPPEARED = 60
MAX_DISTANCE = 80

roi_points = []
display_frame = None
original_frame = None

def resize_video(frame, width=None, height=None):
    if width is None and height is None:
        return frame

    (h, w) = frame.shape[:2]

    if width is not None and height is not None:
        dim = (width, height)
    elif width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = height / float(h)
        dim = (int(w * r), height)

    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized

def get_initial_setup_data():
    """Opens a Tkinter window to let the user select the model and video path."""
    
    # Store the results [model_path, video_path]
    setup_data = [DEFAULT_MODEL_PATH, "test.mp4"] 

    def select_file():
        # Open standard file dialog to choose a video/image file
        filepath = filedialog.askopenfilename(
            title="Select Video or Image File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov"),
                ("Image Files", "*.jpg *.png *.jpeg"),
                ("All Files", "*.*")
            ]
        )
        if filepath:
            video_path_entry.delete(0, tk.END) # Clear current entry
            video_path_entry.insert(0, filepath) # Insert selected path

    def select_and_close():
        # Get the file path from the entry box
        setup_data[1] = video_path_entry.get()
        
        # Get the model path
        selected_key = model_combo.get()
        setup_data[0] = MODEL_OPTIONS.get(selected_key, DEFAULT_MODEL_PATH)
        
        root.destroy()

    root = tk.Tk()
    root.title("Traffic Analyzer Setup")
    
    # Setup window size and centering
    window_width = 450
    window_height = 200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    root.resizable(False, False) # Prevent resizing

    # --- Model Selection Section ---
    label_model = ttk.Label(root, text="1. Choose YOLOv11 Model:")
    label_model.pack(pady=5)

    model_combo = ttk.Combobox(root, values=list(MODEL_OPTIONS.keys()), state="readonly", width=45)
    model_combo.set(list(MODEL_OPTIONS.keys())[0])
    model_combo.pack(pady=2)

    # --- Video Path Section ---
    label_video = ttk.Label(root, text="2. Enter Video Path or Stream URL:")
    label_video.pack(pady=5)

    path_frame = ttk.Frame(root)
    path_frame.pack(padx=10)

    video_path_entry = ttk.Entry(path_frame, width=40)
    video_path_entry.insert(0, setup_data[1]) # Default value
    video_path_entry.pack(side=tk.LEFT, padx=5)
    
    browse_button = ttk.Button(path_frame, text="Browse...", command=select_file)
    browse_button.pack(side=tk.LEFT, padx=5)

    # --- Start Button ---
    select_button = ttk.Button(root, text="Start Analysis", command=select_and_close)
    select_button.pack(pady=10)

    root.mainloop()
    
    return setup_data[0], setup_data[1]


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

# ... (Imports and other functions like get_initial_setup_data remain the same)

def select_roi(first_frame):
    global roi_points, display_frame, original_frame
    
    if first_frame is None:
        print("ERROR in select_roi: first_frame is None.")
        return None
        
    roi_points = []
    original_frame = first_frame.copy()
    display_frame = first_frame.copy()
    
    # --- Window Creation (Modified) ---
    cv2.namedWindow(ROI_WINDOW_NAME, cv2.WINDOW_NORMAL) 
    
    # DISABLED FULLSCREEN HERE
    # cv2.setWindowProperty(ROI_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
    
    cv2.setMouseCallback(ROI_WINDOW_NAME, mouse_callback) 
    
    cv2.imshow(ROI_WINDOW_NAME, display_frame) 
    cv2.waitKey(1) 

    print("Please select 4 points for the congestion zone.")
    # ... (Rest of the function remains the same)
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
    
    # 1. --- INITIAL SETUP ---
    chosen_model_path, video_path_input = get_initial_setup_data()
    MODEL_PATH = chosen_model_path
    VIDEO_PATH = video_path_input

    print(f"Using model: {MODEL_PATH}")
    print(f"Source path: {VIDEO_PATH}")
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    is_image = False

    # Note: This line in your original code was unused/incorrect, so we can ignore or remove it
    # VIDEO = resize_video(VIDEO_PATH, width=800) 
    
    # Determine if input is a local image file
    if isinstance(VIDEO_PATH, str) and not VIDEO_PATH.startswith('http') and not VIDEO_PATH.startswith('rtsp'):
        if os.path.exists(VIDEO_PATH) and os.path.splitext(VIDEO_PATH)[1].lower() in image_extensions:
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
        print(f"Connecting to video stream/file: {VIDEO_PATH}")
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
    
    # --- RESIZE STEP ADDED HERE ---
    # Resize the first frame to 640x800 BEFORE selecting ROI
    # This ensures your coordinate selection matches the processing loop later
    first_frame = resize_video(first_frame, width=640, height=800)
    # ------------------------------

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
    
    # DISABLED FULLSCREEN HERE
    # cv2.setWindowProperty(MAIN_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
            
            # --- RESIZE STEP ADDED HERE ---
            # Resize every incoming frame to the same 640x800 resolution
            if ret:
                current_frame = resize_video(current_frame, width=640, height=800)
            # ------------------------------
            
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