import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def resize_image_short_side(image, image_short_side=736):
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(round(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(round(new_width / width * height / 32) * 32)
    return cv2.resize(image, (new_width, new_height))


# https://github.com/faustomorales/keras-ocr/blob/master/keras_ocr/tools.py
def draw_predictions(raw_image, boxes, scores, figsize=(15, 7)):
    plt.figure(figsize=figsize)
    image = raw_image.copy()
    predictions = sorted(zip(boxes, scores), key=lambda pred: pred[0][:, 1].min())
    left, right = [], []
    
    for box, score in predictions:
        if box[:, 0].min() < image.shape[1] / 2: left.append((box, score))
        else: right.append((box, score))
        cv2.polylines(image, box[np.newaxis], color=(0, 255, 0), thickness=2, isClosed=True)
    plt.imshow(image)
    
    for side, group in zip(['left', 'right'], [left, right]):
        for index, (box, score) in enumerate(group):
            y = 1 - (index / len(group))
            xy = box[0] / np.array([image.shape[1], image.shape[0]])
            xy[1] = 1 - xy[1]
            plt.annotate(
                text = f'{score:.4f}',
                xy = xy,
                xytext = (-0.05 if side == 'left' else 1.05, y),
                xycoords = 'axes fraction',
                arrowprops = {'arrowstyle': '->', 'color': 'r'},
                color = 'r',
                fontsize = 14,
                horizontalalignment = 'right' if side == 'left' else 'left',
            )
    plt.axis('off')
            

class BoxPointsHandler: # Static class
    @staticmethod
    def order_points_clockwise(box_points):
        points = np.array(box_points)
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        quad_box = np.zeros((4, 2), dtype=np.float32)
        quad_box[0] = points[np.argmin(s)]
        quad_box[2] = points[np.argmax(s)]
        quad_box[1] = points[np.argmin(diff)]
        quad_box[3] = points[np.argmax(diff)]
        return quad_box


    @staticmethod
    def get_extremum_points(box_points, image_height, image_width):
        xmin = np.clip(np.floor(box_points[:, 0].min()).astype(np.int32), 0, image_width - 1)
        ymin = np.clip(np.floor(box_points[:, 1].min()).astype(np.int32), 0, image_height - 1)
        xmax = np.clip(np.ceil(box_points[:, 0].max()).astype(np.int32), 0, image_width - 1)
        ymax = np.clip(np.ceil(box_points[:, 1].max()).astype(np.int32), 0, image_height - 1)
        return xmin, ymin, xmax, ymax
    
    
    @staticmethod
    def get_middle_point(point1, point2):
        return (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2
        
    
    @staticmethod
    def get_center_points(text_length, box_points):
        assert len(box_points) == 4
        center_points = []
        
        left_middle_point = BoxPointsHandler.get_middle_point(box_points[0], box_points[3])
        right_middle_point = BoxPointsHandler.get_middle_point(box_points[1], box_points[2])
        
        unit_x = (right_middle_point[0] - left_middle_point[0]) / text_length
        unit_y = (right_middle_point[1] - left_middle_point[1]) / text_length
        
        for i in range(text_length):
            x = left_middle_point[0] + unit_x / 2 + unit_x * i
            y = left_middle_point[1] + unit_y / 2 + unit_y * i
            center_points.append((x, y))
        return center_points
    
    
    @staticmethod
    def get_point_distance(point1, point2):
        dist_x = math.fabs(point1[0] - point2[0])
        dist_y = math.fabs(point1[1] - point2[1])
        return math.sqrt(dist_x**2 + dist_y**2)


    @staticmethod
    def get_diag(box_points):
        diag1 = BoxPointsHandler.get_point_distance(box_points[0], box_points[2])
        diag2 = BoxPointsHandler.get_point_distance(box_points[1], box_points[3])
        return (diag1 + diag2) / 2
