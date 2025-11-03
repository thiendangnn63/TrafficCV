import cv2
import numpy as np
from ultralytics import YOLO
from CentroidTracker import CentroidTracker

class LaneAnalyzer:
    def __init__(self, model_path, roi_points, max_disappeared=60, max_distance=80, conf_threshold=0.3):
        
        self.model = YOLO(model_path)
        self.tracker = CentroidTracker(maxDisappeared=max_disappeared, maxDistance=max_distance)
        
        self.conf_threshold = conf_threshold
        self.vehicle_labels = ['bus', 'motorcycle', 'car', 'truck']
        self.large_vehicle_labels = ['bus', 'car', 'truck']
        self.class_names = self.model.names
        
        self.bboxlab = {}
        self.areaext = np.array(roi_points, dtype=np.int32)
        
        # --- NEW: Calculate Bounding Box for Cropping (minRect) ---
        # Get the coordinates of the smallest upright rectangle that contains the ROI polygon
        x, y, w, h = cv2.boundingRect(self.areaext)
        self.roi_bbox_rect = (x, y, w, h)
        # -----------------------------------------------------------

    def _is_large_vehicle(self, bb):
        if 'large_vehicles' not in self.bboxlab:
            return False
        return bb in self.bboxlab['large_vehicles']

    def process_frame(self, frame):
        
        x_min, y_min, w, h = self.roi_bbox_rect

        # 1. CROP THE FRAME TO THE ROI BOUNDING BOX (FOR FASTER DETECTION)
        cropped_frame = frame[y_min:y_min + h, x_min:x_min + w]
        
        # Run detection on the cropped frame
        bbox_cropped, label, conf = [], [], []
        results = self.model(cropped_frame)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names[cls_id]
                
                if class_name in self.vehicle_labels and confidence > self.conf_threshold:
                    # Coordinates are relative to the cropped frame here
                    x1_c, y1_c, x2_c, y2_c = box.xyxy[0]
                    
                    # 2. TRANSLATE COORDINATES BACK TO ORIGINAL FRAME
                    x1_f = int(x1_c + x_min)
                    y1_f = int(y1_c + y_min)
                    x2_f = int(x2_c + x_min)
                    y2_f = int(y2_c + y_min)
                    
                    b = (x1_f, y1_f, x2_f, y2_f)
                    
                    bbox_cropped.append(b)
                    label.append(class_name)
                    conf.append(confidence)

        # Update the 'large_vehicles' list for this frame using the full-frame coordinates
        self.bboxlab['large_vehicles'] = [bbox_cropped[i] for i, x in enumerate(label) if x in self.large_vehicle_labels]
        
        objects = self.tracker.update(bbox_cropped)

        current_congestion_score = 0
        
        outputimage = frame.copy()
        overlay = outputimage.copy()

        # Draw the transparent ROI fill
        roi_color = (0, 255, 0)
        alpha = 0.3
        beta = 1.0 - alpha
        cv2.fillPoly(overlay, [self.areaext], roi_color)
        outputimage = cv2.addWeighted(overlay, alpha, outputimage, beta, 0)

        # Draw the ROI outline
        cv2.polylines(outputimage, [self.areaext], isClosed=True, color=roi_color, thickness=2)
        
        # Draw detection boxes (using the translated full-frame coordinates)
        for (box, lbl) in zip(bbox_cropped, label):
            x1, y1, x2, y2 = box
            cv2.rectangle(outputimage, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(outputimage, lbl, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw tracker IDs and calculate score
        for (objid, bb) in objects.items():
            if self.tracker.disappeared[objid] > 0:
                continue

            x1, y1, x2, y2 = bb
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)

            text = f"ID: {objid}"
            cv2.putText(outputimage, text, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Check if object is in the congestion zone
            res = cv2.pointPolygonTest(self.areaext, (cx, cy), False)
            
            if res >= 0.0:
                if self._is_large_vehicle(bb):
                    current_congestion_score += 3
                else:
                    current_congestion_score += 1

        cv2.putText(outputimage, f"Congestion Score: {current_congestion_score}", (5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return current_congestion_score, outputimage
