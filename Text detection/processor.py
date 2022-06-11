import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon
from utils import BoxPointsHandler


class PostProcessor:
    def __init__(self, thresh=0.3, min_box_score=0.7, max_candidates=500, unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.min_box_score = min_box_score
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        

    def __call__(self, binarize_map, batch_true_sizes, output_polygon=False):
        segmentation = binarize_map > self.thresh
        batch_boxes, batch_scores = [], []
        
        for batch_idx, image_size in enumerate(batch_true_sizes):
            height, width = image_size
            convert_func = self.bitmap2polys if output_polygon else self.bitmap2quads
            boxes_in_image, scores = convert_func(binarize_map[batch_idx], segmentation[batch_idx], width, height)
            
            batch_boxes.append(boxes_in_image)
            batch_scores.append(scores)
        return batch_boxes, batch_scores
    

    def bitmap2polys(self, pred, bitmap, original_width, original_height):
        assert len(bitmap.shape) == 2
        boxes, scores = [], []
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4: continue

            score = self.box_score_fast(pred, contour.reshape(-1, 2))
            if self.min_box_score > score: continue

            if points.shape[0] > 2:
                box = self.unclip(points)
                if len(box) > 1: continue
            else: continue
            
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2: continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * original_width), 0, original_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * original_height), 0, original_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores


    def bitmap2quads(self, pred, bitmap, original_width, original_height):
        assert len(bitmap.shape) == 2
        boxes, scores = [], []
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size: continue
            
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.min_box_score > score: continue

            box = self.unclip(points)
            box, sside = self.get_mini_boxes(box.reshape(-1, 1, 2))
            if sside < self.min_size + 2: continue
            
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * original_width), 0, original_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * original_height), 0, original_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores


    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded


    def get_mini_boxes(self, contour):
        try: bounding_box = cv2.minAreaRect(contour)
        except: return [], 0
        
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        
        if points[1][1] > points[0][1]: index_1, index_4 = 0, 1
        else: index_1, index_4 = 1, 0
            
        if points[3][1] > points[2][1]: index_2, index_3 = 2, 3
        else: index_2, index_3 = 3, 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])


    def box_score_fast(self, bitmap, box):
        h, w = bitmap.shape[:2]
        xmin, ymin, xmax, ymax = BoxPointsHandler.get_extremum_points(box, h, w)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        
        new_box = box.copy()
        new_box[:, 0] = new_box[:, 0] - xmin
        new_box[:, 1] = new_box[:, 1] - ymin
        
        cv2.fillPoly(mask, new_box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
