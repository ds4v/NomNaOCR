import cv2
import ast
import json
import pyclipper
import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa

from pathlib import Path
from shapely.geometry import Polygon
from transform import transform, crop, resize
np.random.seed(2022)


class DataImporter:
    def __init__(self, dataset_dir, pattern):
        self.img_paths, self.all_bboxes = [], []
        self.bboxes_count = 0

        for path in Path(dataset_dir).rglob(pattern):
            lines = open(path, encoding='utf-8').read()
            lines = lines.replace(path.parent.name, str(path.parent))

            for line in lines.rstrip('\n').split('\n'):
                image_path, image_bboxes = line.split('\t')
                try: image_bboxes = ast.literal_eval(image_bboxes)
                except: image_bboxes = json.loads(image_bboxes)
                image_bboxes = [bbox for bbox in image_bboxes if len(bbox['points']) >= 3]
                
                self.img_paths.append(image_path)
                self.all_bboxes.append(image_bboxes)
                self.bboxes_count += len(image_bboxes)
            

    def split(self, ratio=0.8):
        train_size = int(len(self.img_paths) * ratio)
        train_img_paths, valid_img_paths = self.img_paths[:train_size], self.img_paths[train_size:]
        all_train_bboxes, all_valid_bboxes = self.all_bboxes[:train_size], self.all_bboxes[train_size:]
        return train_img_paths, all_train_bboxes, valid_img_paths, all_valid_bboxes


    def __str__(self):
        return (
            f'Samples count (1 image can have multiple bounding boxes):'
            f'\n- Number of images found: {len(self.img_paths)}'
            f'\n- Number of image bounding boxes: {len(self.all_bboxes)}'
            f'\n- Number of bounding boxes in all images: {self.bboxes_count}'
        )


class DBNetDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self, img_paths=[], all_bboxes=[], batch_size=16, img_size=640,
        thresh_min=0.3, thresh_max=0.7, shrink_ratio=0.4, is_training=True
    ):
        self.img_paths = img_paths
        self.all_bboxes = all_bboxes
        self.batch_size = batch_size
        self.img_size = img_size

        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.shrink_ratio = shrink_ratio
        self.is_training = is_training

        self.transform_aug = iaa.Sequential([iaa.Affine(rotate=(-10, 10)), iaa.Resize((0.5, 3.0))])
        self.size = len(self.img_paths)
        self.mean = [103.939, 116.779, 123.68]
        self.on_epoch_end()


    def _init_inputs(self):
        batch_images = np.zeros([self.batch_size, self.img_size, self.img_size, 3], dtype=np.float32)
        batch_gts = np.zeros([self.batch_size, self.img_size, self.img_size], dtype=np.float32)
        batch_masks = np.zeros([self.batch_size, self.img_size, self.img_size], dtype=np.float32)
        batch_thresh_maps = np.zeros([self.batch_size, self.img_size, self.img_size], dtype=np.float32)
        batch_thresh_masks = np.zeros([self.batch_size, self.img_size, self.img_size], dtype=np.float32)
        return batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks


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


    def _draw_thresh_map(self, points, canvas, mask, shrink_ratio=0.4):
        points = np.array(points)
        polygon = Polygon(points)
        assert points.ndim == 2 and points.shape[-1] == 2 and polygon.is_valid

        padding = pyclipper.PyclipperOffset()
        padding.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        distance = polygon.area * (1 - shrink_ratio**2) / polygon.length
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
        canvas[ymin_valid:ymax_valid, xmin_valid:xmax_valid] = np.fmax(1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymin,
            xmin_valid - xmin:xmax_valid - xmin
        ], canvas[ymin_valid:ymax_valid, xmin_valid:xmax_valid])


    def __getitem__(self, batch_idx): # Generate 1 batch of data
        indexes_in_batch = self.indexes[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
        batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks = self._init_inputs()

        for batch_count, image_idx in enumerate(indexes_in_batch):
            image = cv2.imread(self.img_paths[image_idx])
            image_bboxes = self.all_bboxes[image_idx]

            if self.is_training:
                transform_aug = self.transform_aug.to_deterministic()
                image, image_bboxes = transform(transform_aug, image, image_bboxes)
                image, image_bboxes = crop(image, image_bboxes)
            image, image_bboxes = resize(self.img_size, image, image_bboxes)

            gt = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            mask = np.ones((self.img_size, self.img_size), dtype=np.float32)
            thresh_map = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            thresh_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

            for bbox in image_bboxes: # Generate gt and mask
                polygon = Polygon(bbox['points'])
                if not polygon.is_valid: continue

                points = np.array(bbox['points'])
                if polygon.area < 1:
                    cv2.fillPoly(mask, points.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    distance = polygon.area * (1 - self.shrink_ratio**2) / polygon.length

                    shrinked = padding.Execute(-distance)
                    if len(shrinked) == 0:
                        cv2.fillPoly(mask, points.astype(np.int32)[np.newaxis, :, :], 0)
                        continue
                    else:
                        shrinked = np.array(shrinked[0]).reshape(-1, 2)
                        if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
                            cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                        else:
                            cv2.fillPoly(mask, points.astype(np.int32)[np.newaxis, :, :], 0)
                            continue

                # Generate thresh map and thresh mask
                self._draw_thresh_map(bbox['points'], thresh_map, thresh_mask, self.shrink_ratio)

            batch_images[batch_count] = image.astype(np.float32) - self.mean
            batch_gts[batch_count] = gt
            batch_masks[batch_count] = mask
            batch_thresh_maps[batch_count] = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min
            batch_thresh_masks[batch_count] = thresh_mask
        return [batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks], []


    def __len__(self):
        return np.ceil(self.size / self.batch_size).astype(int)


    def on_epoch_end(self):
        self.indexes = np.arange(self.size)
        if self.is_training: 
            np.random.shuffle(self.indexes)
