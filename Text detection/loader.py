import cv2
import copy
import pyclipper
import numpy as np
import tensorflow as tf
from shapely.geometry import Polygon
from utils import BoxPointsHandler


class AnnotationsImporter:
    def __init__(self, paths_map):
        self.annotations = []
        self.images_count = 0
        self.all_boxes_count = 0

        with open(paths_map, encoding='utf-8') as file:
            for line in file:
                image_path, gts_path = line.rstrip('\n').split('\t')
                boxes_in_image, texts_in_image = self.parse_gts(gts_path)

                self.images_count += 1
                self.annotations.append({
                    'image_path': image_path,
                    'quad_boxes': boxes_in_image,
                    'texts': texts_in_image,
                })
                print(f'[GET] Loading from {paths_map}: {self.images_count} images', end='\r', flush=True)
            print()
    

    def parse_gts(self, gts_path):
        with open(gts_path, encoding='utf-8') as file:
            boxes, texts = [], []
            for idx, line in enumerate(file):
                annotation = line.rstrip('\n').replace(u'\ufeff', '').split(',') # x1,y1,x2,y2,x3,y3,x4,y4,text
                try: 
                    box_points = BoxPointsHandler.order_points_clockwise([
                        (float(annotation[0]), float(annotation[1])), # (x1, y1)
                        (float(annotation[2]), float(annotation[3])), # (x2, y2)
                        (float(annotation[4]), float(annotation[5])), # (x3, y3)
                        (float(annotation[6]), float(annotation[7]))  # (x4, y4)
                    ])
                    self.all_boxes_count += 1
                    boxes.append(box_points)
                    texts.append(annotation[8])
                except Exception as err:
                    print(f'Line {idx + 1} of {gts_path}: \n{err} => skip this bounding box')
                    continue
        return boxes, texts


class DBNetGenerator(tf.keras.utils.Sequence):
    def __init__(
        self, annotations, batch_size=16, image_size=640, ignore_texts=['###'], 
        thresh_min=0.3, thresh_max=0.7, shrink_ratio=0.4, seed=None
    ):
        self.annotations = copy.deepcopy(annotations)
        self.batch_size = batch_size
        self.image_size = image_size
        self.ignore_texts = ignore_texts

        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.shrink_ratio = shrink_ratio
        self.seed = seed
        
        if seed: np.random.seed(seed) # If a seed was given, use it to shuffle the data
        self.size = len(self.annotations)
        self.on_epoch_end()

    
    def __getitem__(self, batch_idx): # Generate 1 batch of data
        indexes_in_batch = self.indexes[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
        batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks = self._init_inputs()

        for batch_count, image_idx in enumerate(indexes_in_batch):
            image_annotations = copy.deepcopy(self.annotations[image_idx])
            image = cv2.imread(image_annotations['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            image, image_annotations = self._resize(image, image_annotations)

            gt = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
            thresh_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            thresh_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            for box_idx, box_points in enumerate(image_annotations['quad_boxes']): # Generate gt and mask
                polygon = Polygon(box_points)
                if not polygon.is_valid: continue

                points = np.array(box_points)
                if polygon.area < 1 or image_annotations['texts'][box_idx] in self.ignore_texts:
                    cv2.fillPoly(mask, points.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                
                distance = polygon.area * (1 - self.shrink_ratio**2) / polygon.length
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                
                if len(shrinked) == 0:
                    cv2.fillPoly(mask, points.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
                    cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                    self._draw_thresh_map(box_points, thresh_map, thresh_mask) # Generate thresh map and thresh mask
                else:
                    cv2.fillPoly(mask, points.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
               
            batch_images[batch_count] = image.astype(np.float32) / 255.0
            batch_gts[batch_count] = gt
            batch_masks[batch_count] = mask
            batch_thresh_maps[batch_count] = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min
            batch_thresh_masks[batch_count] = thresh_mask
        return batch_images, [batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks]

    
    def _init_inputs(self):
        batch_images = np.zeros([self.batch_size, self.image_size, self.image_size, 3], dtype=np.float32)
        batch_gts = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
        batch_masks = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
        batch_thresh_maps = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
        batch_thresh_masks = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
        return batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks

    
    def _resize(self, image, image_annotations):
        h, w, c = image.shape
        scale = min(self.image_size / w, self.image_size / h)
        h, w = int(h * scale), int(w * scale)
        
        pad_image = np.zeros((self.image_size, self.image_size, c), image.dtype)
        pad_image[:h, :w] = cv2.resize(image, (w, h))
        image_annotations['quad_boxes'] = np.array(image_annotations['quad_boxes']).astype(np.float64) * scale
        return pad_image, image_annotations
    
    
    def _compute_distance(self, xs, ys, point_1, point_2):
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) /\
                (2 * np.sqrt(square_distance_1 * square_distance_2) + 1e-6)
        square_sin = np.nan_to_num(1 - np.square(cosin))

        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / (square_distance + 1e-6))
        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        return result


    def _draw_thresh_map(self, box, canvas, mask):
        points = np.array(box)
        polygon = Polygon(points)
        assert points.ndim == 2 and points.shape[-1] == 2 and polygon.is_valid

        distance = polygon.area * (1 - self.shrink_ratio**2) / polygon.length
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin, xmax = padded_polygon[:, 0].min(), padded_polygon[:, 0].max()
        ymin, ymax = padded_polygon[:, 1].min(), padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        points[:, 0] = points[:, 0] - xmin
        points[:, 1] = points[:, 1] - ymin
        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros((points.shape[0], height, width), dtype=np.float32)
        for i in range(points.shape[0]):
            j = (i + 1) % points.shape[0]
            absolute_distance = self._compute_distance(xs, ys, points[i], points[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

        distance_map = np.min(distance_map, axis=0)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymax + height,
            xmin_valid - xmin:xmax_valid - xmax + width
        ], canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
        

    def __len__(self):
        return np.ceil(self.size / self.batch_size).astype(np.int32)


    def on_epoch_end(self):
        self.indexes = np.arange(self.size)
        if self.seed: np.random.shuffle(self.indexes)
