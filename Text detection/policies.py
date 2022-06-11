import math
import imagesize
import numpy as np
from shapely.geometry import Polygon
from utils import BoxPointsHandler


def scoring_policy_compute(image_annotations, ignore_texts):
    true_polys = [] # List of shapely.geometry.Polygon
    true_boxes = [] # List of numpy boxes
    true_boxes_pccs = [] # List of Pseudo character centers points
    true_ignore_idxs = [] # List of Ground Truth Polygons' keys marked as ignore
    
    for idx, box in enumerate(image_annotations['quad_boxes']):
        text_in_box = image_annotations['texts'][idx]
        true_polys.append(Polygon(box))
        
        if text_in_box in ignore_texts:
            true_boxes.append(box)
            true_boxes_pccs.append([])
            true_ignore_idxs.append(idx)
        else:
            text_length = len(text_in_box)
            width, height = imagesize.get(image_annotations['image_path'])
            xmin, ymin, xmax, ymax = BoxPointsHandler.get_extremum_points(box, height, width)
            aspect_ratio = (ymax - ymin) / (xmax - xmin)

            box = [box[3], box[0], box[1], box[2]] if aspect_ratio > 1.5 else box
            box_center_points = BoxPointsHandler.get_center_points(text_length, box)
            true_boxes.append(box)
            true_boxes_pccs.append(box_center_points)

    # GT Don't Care overlap
    true_keep_idxs = list(set(range(len(true_polys))) - set(true_ignore_idxs))
    for ignore_idx in true_ignore_idxs:
        for keep_idx in true_keep_idxs:
            if true_polys[keep_idx].intersection(true_polys[ignore_idx]).area > 0:
                true_polys[ignore_idx] -= true_polys[keep_idx]
    return true_polys, true_boxes, true_boxes_pccs, true_ignore_idxs
    

class MatchingPolicy:
    def __init__(
        self, true_polys, pred_polys, 
        true_ignore_idxs, pred_ignore_idxs, 
        precision_matrix, recall_matrix, 
        area_precision_constraint, area_recall_constraint
    ):
        self.true_polys = true_polys
        self.true_ignore_idxs = true_ignore_idxs
        self.true_exclude_matrix = np.zeros(len(true_polys), np.int8)
        
        self.pred_polys = pred_polys
        self.pred_ignore_idxs = pred_ignore_idxs
        self.pred_exclude_matrix = np.zeros(len(pred_polys), np.int8)
        
        self.precision_matrix = precision_matrix
        self.recall_matrix = recall_matrix
        self.area_precision_constraint = area_precision_constraint
        self.area_recall_constraint = area_recall_constraint
        
    
    def _get_angle_3pt(self, a, b, c):
        angle = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        angle = math.degrees(angle)
        return angle + 360 if angle < 0 else angle
    
    
    def _get_pivots(self, rects, polys):
        pivots = []
        for match_idx in rects:
            box_points = list(polys[match_idx].exterior.coords)
            middle_point = BoxPointsHandler.get_middle_point(box_points[0], box_points[3])
            pivots.append([middle_point, polys[match_idx].centroid.coords[0]])
        return pivots
        
    
    def one2one(self, row, col):
        count = 0
        for i in range(len(self.recall_matrix[0])):
            if self.precision_matrix[row, i] >= self.area_precision_constraint and \
                self.recall_matrix[row, i] >= self.area_recall_constraint: count += 1
        if count != 1: return False
        
        count = 0
        for i in range(len(self.recall_matrix)):
            if self.precision_matrix[i, col] >= self.area_precision_constraint and \
                self.recall_matrix[i, col] >= self.area_recall_constraint: count += 1
        if count != 1: return False
        
        if self.precision_matrix[row, col] >= self.area_precision_constraint and \
            self.recall_matrix[row, col] >= self.area_recall_constraint: return True
        return False
   
    
    def one2many(self, true_idx):
        many_sum, pred_rects = 0, []
        for pred_idx in range(len(self.recall_matrix[0])):
            if pred_idx not in self.pred_ignore_idxs and \
                self.true_exclude_matrix[true_idx] == 0 and \
                self.pred_exclude_matrix[pred_idx] == 0 and \
                self.precision_matrix[true_idx, pred_idx] >= self.area_precision_constraint:
                many_sum += self.recall_matrix[true_idx, pred_idx]
                pred_rects.append(pred_idx)
                    
        if many_sum >= self.area_recall_constraint and len(pred_rects) >= 2:
            pivots = self._get_pivots(pred_rects, self.pred_polys)
            pivots_length = len(pivots)
            
            for i in range(pivots_length):
                for j in range(pivots_length):
                    if i != j:
                        angle = self._get_angle_3pt(pivots[i][0], pivots[j][1], pivots[i][1])
                        if angle > 180: angle = 360 - angle
                        if min(angle, 180 - angle) >= 45: return False, []
            return True, pred_rects
        return False, []
    
    
    def many2one(self, pred_idx):
        many_sum, true_rects = 0, []
        for true_idx in range(len(self.recall_matrix)):
            if true_idx not in self.true_ignore_idxs and \
                self.true_exclude_matrix[true_idx] == 0 and \
                self.pred_exclude_matrix[pred_idx] == 0 and \
                self.recall_matrix[true_idx, pred_idx] >= self.area_recall_constraint:
                many_sum += self.precision_matrix[true_idx, pred_idx]
                true_rects.append(true_idx)
                    
        if many_sum >= self.area_precision_constraint and len(true_rects) >= 2:
            pivots = self._get_pivots(true_rects, self.true_polys)
            pivots_length = len(pivots)
            
            for i in range(pivots_length):
                for j in range(pivots_length):
                    if i != j:
                        angle = self._get_angle_3pt(pivots[i][0], pivots[j][1], pivots[i][1])
                        if angle > 180: angle = 360 - angle
                        if min(angle, 180 - angle) >= 45: return False, []
            return True, true_rects
        return False, []