import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon


def resize(size, image, image_bboxes):
    h, w, c = image.shape
    scale = min(size / w, size / h)
    h, w = int(h * scale), int(w * scale)
    pad_image = np.zeros((size, size, c), image.dtype)
    pad_image[:h, :w] = cv2.resize(image, (w, h))

    new_bboxes = []
    for bbox in image_bboxes:
        points = np.array(bbox['points']).astype(np.float64) * scale
        new_bboxes.append({
            'transcription': bbox['transcription'], 
            'points': points.tolist(), 
            'difficult': bbox['difficult']
        })
    return pad_image, new_bboxes


def compute_distance(xs, ys, point_1, point_2):
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) /\
            (2 * np.sqrt(square_distance_1 * square_distance_2) + 1e-6)
    square_sin = np.nan_to_num(1 - np.square(cosin))

    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / (square_distance + 1e-6))
    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result


def draw_thresh_map(points, canvas, mask, shrink_ratio=0.4):
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
        absolute_distance = compute_distance(xs, ys, points[i], points[j])
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