# https://github.com/clovaai/TedEval
import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from shapely.geometry import Polygon, Point
from utils import resize_image_short_side, BoxPointsHandler
from policies import scoring_policy_compute, MatchingPolicy


class TedEvalMetric(tf.keras.callbacks.Callback):
    def __init__(
        self, true_annotations, ignore_texts=['###'], min_box_score=0.7, image_short_side=736,
        area_precision_constraint=0.4, area_recall_constraint=0.4, progressbar=tqdm, eval_steps=1
    ):
        super().__init__()
        self.true_annotations = true_annotations
        self.ignore_texts = ignore_texts
        self.min_box_score = min_box_score
        
        self.area_precision_constraint = area_precision_constraint
        self.area_recall_constraint = area_recall_constraint
        self.progressbar = progressbar 
        self.eval_steps = eval_steps
        
        self.images_and_sizes = []
        for image_annotations in self.progressbar(self.true_annotations, unit='image', desc='Reading evaluation images'):
            image = cv2.imread(image_annotations['image_path'])
            true_size = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            image = resize_image_short_side(image, image_short_side=image_short_side)
            image = image.astype(np.float32) / 255.0
            self.images_and_sizes.append((image, true_size))
    
    
    def on_train_begin(self, logs=None):
        self.best_epoch = 0
        self.best_loss = np.Inf  # Initialize the best loss as infinity
        self.best_weights = None # Store the weights at which the minimum loss occurs
        
        
    def on_train_end(self, logs=None):
        print('\nFinal evaluation with the best weights from epoch:', self.best_epoch + 1)
        self.model.set_weights(self.best_weights)
        self._gather_measures()
        
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if np.less(current_loss, self.best_loss):
            self.best_epoch = epoch
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights() # Record the best weights if current results is better (less).
        
        if (epoch + 1) % self.eval_steps == 0:
            self._gather_measures()
        
        
    def _gather_measures(self):
        self.all_pred_boxes = []
        self.mean_precision = 0
        self.mean_recall = 0
        self.mean_hmean = 0
        self.epoch_true_count = 0
        self.epoch_pred_count = 0
        
        for image, true_size in self.progressbar(self.images_and_sizes, unit='image', desc='Predicting bounding boxes'):
            batch_boxes, batch_scores = self.model.predict(tf.expand_dims(image, 0), [true_size])
            self.all_pred_boxes.append([
                box for idx, box in enumerate(batch_boxes[0]) # Remove batch dimension
                if batch_scores[0][idx] > self.min_box_score
            ])

        iterator = self.progressbar(
            zip(self.true_annotations, self.all_pred_boxes), 
            total = len(self.true_annotations), 
            unit = 'image',
            desc = 'Calculating TedEval metric',
        )
        
        for image_annotations, image_pred_boxes in iterator:
            precision, recall, true_care_count, pred_care_count = self.get_metrics(image_annotations, image_pred_boxes)
            self.mean_precision += precision
            self.mean_recall += recall
            self.epoch_true_count += true_care_count
            self.epoch_pred_count += pred_care_count
        
        self.mean_precision = 0 if self.epoch_pred_count == 0 else self.mean_precision / self.epoch_pred_count
        self.mean_recall = 0 if self.epoch_true_count == 0 else self.mean_recall / self.epoch_true_count
        if self.mean_precision + self.mean_recall == 0: self.mean_hmean = 0
        else: self.mean_hmean = 2 * self.mean_precision * self.mean_recall / (self.mean_precision + self.mean_recall)
        
        print(f'Average metrics for all evaluation images '
              f'- precision: {self.mean_precision:.4f} '
              f'- recall: {self.mean_recall:.4f} '
              f'- hmean: {self.mean_hmean:.4f}')
    
    
    def get_metrics(self, annotations, pred_boxes):
        true_polys, true_boxes, true_boxes_pccs, true_ignore_idxs = scoring_policy_compute(annotations, self.ignore_texts)
        pred_polys, pred_ignore_idxs = [Polygon(box) for box in pred_boxes], []
        true_polys_length, pred_polys_length = len(true_polys), len(pred_polys) 
        precision = recall = 0.0
        
        if true_polys_length > 0 and pred_polys_length > 0:
            precision_mat, recall_mat, true_pccs_mat = self._get_metrics_matrices(true_boxes_pccs, true_polys, pred_polys)
            pred_ignore_idxs, pred_polys = self._compute_pred_ignores(
                pred_polys, true_polys, true_ignore_idxs, 
                precision_mat, recall_mat
            )
            
            # Recalculate matrices
            for true_idx, true_poly in enumerate(true_polys):
                for pred_idx, pred_poly in enumerate(pred_polys):
                    intersected_area = pred_poly.intersection(true_poly).area 
                    precision_mat[true_idx, pred_idx] = 0 if pred_poly.area == 0 else intersected_area / pred_poly.area
                    recall_mat[true_idx, pred_idx] = 0 if true_poly.area == 0 else intersected_area / true_poly.area

            '''
            Apply matching policy:
            - Non-exclusively gathers all possible matches of not only one-to-one but also one-to-many and many-to-one.
            - The threshold of both area recall and area precision are set to 0.4.
            - Multiline is identified and rejected when |min(theta, 180 - theta)| > 45
            '''
            matching_policy = MatchingPolicy(
                true_polys, pred_polys, true_ignore_idxs, pred_ignore_idxs, 
                precision_mat, recall_mat, self.area_precision_constraint, self.area_recall_constraint
            )
            pairs = []
        
            # Find many-to-one matches
            for pred_idx in range(len(pred_polys)):
                if pred_idx not in pred_ignore_idxs:
                    is_match, true_matches = matching_policy.many2one(pred_idx)
                    if is_match: pairs.append({'true': true_matches, 'pred': [pred_idx]})

            # Find one-to-one matches
            for true_idx, true_poly in enumerate(true_polys):
                for pred_idx, pred_poly in enumerate(pred_polys):
                    if true_idx not in true_ignore_idxs and \
                        pred_idx not in pred_ignore_idxs and \
                        matching_policy.one2one(true_idx, pred_idx):
                        norm_dist = BoxPointsHandler.get_point_distance(
                            true_poly.centroid.coords[0], 
                            pred_poly.centroid.coords[0]
                        ) / (
                            BoxPointsHandler.get_diag(true_boxes[true_idx]) + 
                            BoxPointsHandler.get_diag(pred_boxes[pred_idx])
                        ) * 2.0
                        if norm_dist < 1: pairs.append({'true': [true_idx], 'pred': [pred_idx]})
                        
            # Find one-to-many matches
            for true_idx in range(len(true_polys)):
                if true_idx not in true_ignore_idxs:
                    is_match, pred_matches = matching_policy.one2many(true_idx)
                    if is_match: pairs.append({'true': [true_idx], 'pred': pred_matches})
            
            # Fill the match matrix
            match_mat = np.zeros([true_polys_length, pred_polys_length])
            for pair in pairs: match_mat[pair['true'], pair['pred']] = 1
            
            # Fill the character matrix
            for pred_idx in np.where(match_mat.sum(axis=0) > 0)[0]:
                for true_idx, true_box_pccs in enumerate(true_boxes_pccs):
                    if match_mat[true_idx, pred_idx] != 1: continue
                    for pcc_idx, center_point in enumerate(true_box_pccs):
                        if pred_polys[pred_idx].contains(Point(center_point)):
                            true_pccs_mat[true_idx][pred_idx][pcc_idx] = 1
                        
            # Compute precision and recall
            precision, recall = self._compute_precision_recall(
                match_mat, true_pccs_mat, true_boxes_pccs, 
                true_polys_length, pred_polys_length
            )

        true_care_count = len(true_polys) - len(true_ignore_idxs)
        pred_care_count = len(pred_polys) - len(pred_ignore_idxs)
        return precision, recall, true_care_count, pred_care_count
    
    
    def _get_metrics_matrices(self, true_boxes_pccs, true_polys, pred_polys):
        true_polys_length, pred_polys_length = len(true_polys), len(pred_polys)
        precision_mat = np.empty([true_polys_length, pred_polys_length])
        recall_mat = np.empty([true_polys_length, pred_polys_length])
        true_pccs_mat = []
        
        for true_idx, true_poly in enumerate(true_polys):
            rows = []
            for pred_idx, pred_poly in enumerate(pred_polys):
                intersected_area = pred_poly.intersection(true_poly).area 
                precision_mat[true_idx, pred_idx] = 0 if pred_poly.area == 0 else intersected_area / pred_poly.area
                recall_mat[true_idx, pred_idx] = 0 if true_poly.area == 0 else intersected_area / true_poly.area
                rows.append(np.zeros(len(true_boxes_pccs[true_idx])))
            true_pccs_mat.append(rows)
        return precision_mat, recall_mat, true_pccs_mat
    
    
    def _compute_pred_ignores(self, pred_polys, true_polys, true_ignore_idxs, precision_mat, recall_mat):
        if len(true_ignore_idxs) <= 0: return [], pred_polys
        pred_ignore_idxs = [] # List of Detected Polygons' matched marked as ignore
        
        for pred_idx in range(len(pred_polys)):
            # Many-to-one
            many_sum = 0
            for ignore_idx in true_ignore_idxs:
                if recall_mat[ignore_idx, pred_idx] > self.area_recall_constraint: 
                    many_sum += precision_mat[ignore_idx, pred_idx]
                
            if many_sum >= self.area_precision_constraint: 
                pred_ignore_idxs.append(pred_idx)
            else:
                for ignore_idx in true_ignore_idxs:
                    if precision_mat[ignore_idx, pred_idx] > self.area_precision_constraint:
                        pred_ignore_idxs.append(pred_idx)
                        break
                    
            # Many-to-one for mixed ignore and not ignore
            for ignore_idx in true_ignore_idxs:
                if recall_mat[ignore_idx, pred_idx] > 0: 
                    pred_polys[pred_idx] -= true_polys[ignore_idx]
        return pred_ignore_idxs, pred_polys
    
    
    def _compute_precision_recall(self, match_mat, true_pccs_mat, true_boxes_pccs, true_length, pred_length):
        true_rect_mat = np.zeros(true_length, np.int8)
        pred_rect_mat = np.zeros(pred_length, np.int8)
        precision = recall = 0.0
        
        for pred_idx in range(len(pred_rect_mat)):
            if match_mat.sum(axis=0)[pred_idx] > 0:
                count = total = 0
                for true_idx in range(len(true_rect_mat)):
                    if match_mat[true_idx, pred_idx] > 0:
                        count += len(np.where(true_pccs_mat[true_idx][pred_idx] == 1)[0])
                        total += len(true_pccs_mat[true_idx][pred_idx])
                precision += count / total
        
        for true_idx in range(len(true_rect_mat)):
            if match_mat.sum(axis=1)[true_idx] > 0: 
                count = len(np.where(sum(true_pccs_mat[true_idx]) == 1)[0])
                total = len(true_boxes_pccs[true_idx])
                recall += count / total
        return precision, recall
