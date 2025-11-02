import cv2
import numpy as np
from LaneAnalyzer import LaneAnalyzer
import time

MODEL_PATH = "yolov10n.pt"
VIDEO_PATH = "test.mp4"
ROI_WINDOW_NAME = "Select Congestion Zone"

roi_points = []
display_frame = None
original_frame = None

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
    
    cv2.namedWindow(ROI_WINDOW_NAME)
    cv2.setMouseCallback(ROI_WINDOW_NAME, mouse_callback)
    cv2.moveWindow(ROI_WINDOW_NAME, 100, 100)
    
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
    print("Frame grabbed. Please select ROI.")

    user_roi_points = select_roi(first_frame)

    if user_roi_points is None:
        print("Setup cancelled. Exiting.")
        video.release()
        cv2.destroyAllWindows()
        return

    analyzer = LaneAnalyzer(model_path=MODEL_PATH, roi_points=user_roi_points)
    start_time = time.time()
    frame_count = 0
    print("Starting traffic analysis...")

    while True:
        if frame_count == 0:
            current_frame = first_frame
        else:
            ret, current_frame = video.read()
            if not ret:
                print("Stream ended or frame could not be read.")
                break
        
        score, annotated_frame = analyzer.process_frame(current_frame)
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (5, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Traffic Congestion Analyzer", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
