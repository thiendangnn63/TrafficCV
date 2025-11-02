import cv2
import numpy as np
from ultralytics import YOLO
from CentroidTracker import CentroidTracker

class LaneAnalyzer:
    def __init__(self, model_path, roi_points, max_disappeared=50, max_distance=50, conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.tracker = CentroidTracker(maxDisappeared=max_disappeared, maxDistance=max_distance)
        
        self.conf_threshold = conf_threshold
        self.vehicle_labels = ['bus', 'motorcycle', 'car', 'truck']
        self.large_vehicle_labels = ['bus', 'car', 'truck']
        self.class_names = self.model.names
        
        self.bboxlab = {}
        
        self.areaext = np.array(roi_points, dtype=np.int32)

    def _is_large_vehicle(self, bb):
        if 'large_vehicles' not in self.bboxlab:
            return False
        return bb in self.bboxlab['large_vehicles']

    def process_frame(self, frame):
        
        bbox, label, conf = [], [], []
        results = self.model(frame)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names[cls_id]
                
                if class_name in self.vehicle_labels and confidence > self.conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0]
                    b = (int(x1), int(y1), int(x2), int(y2))
                    bbox.append(b)
                    label.append(class_name)
                    conf.append(confidence)

        self.bboxlab['large_vehicles'] = [bbox[i] for i, x in enumerate(label) if x in self.large_vehicle_labels]
        
        objects = self.tracker.update(bbox)

        current_congestion_score = 0
        
        outputimage = frame.copy()
        
        overlay = outputimage.copy()
        
        roi_color = (0, 255, 0)
        alpha = 0.3
        beta = 1.0 - alpha
        
        cv2.fillPoly(overlay, [self.areaext], roi_color)
        
        outputimage = cv2.addWeighted(overlay, alpha, outputimage, beta, 0)

        cv2.polylines(outputimage, [self.areaext], isClosed=True, color=roi_color, thickness=2)
        
        for (box, lbl) in zip(bbox, label):
            x1, y1, x2, y2 = box
            cv2.rectangle(outputimage, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(outputimage, lbl, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for (objid, bb) in objects.items():
            
            if self.tracker.disappeared[objid] > 0:
                continue
            
            x1, y1, x2, y2 = bb
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)

            text = f"ID: {objid}"
            cv2.putText(outputimage, text, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            res = cv2.pointPolygonTest(self.areaext, (cx, cy), False)
            
            if res >= 0.0:
                if self._is_large_vehicle(bb):
                    current_congestion_score += 3
                else:
                    current_congestion_score += 1

        cv2.putText(outputimage, f"Congestion Score: {current_congestion_score}", (5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return current_congestion_score, outputimage
