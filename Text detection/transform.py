import cv2
import imgaug
import numpy as np


def transform(aug, image, image_bboxes):
    image_shape = image.shape
    image = aug.augment_image(image)
    new_bboxes = []

    for bbox in image_bboxes:
        keypoints = aug.augment_keypoints([
            imgaug.KeypointsOnImage(
                [imgaug.Keypoint(point[0], point[1]) for point in bbox['points']], 
                shape = image_shape
            )
        ])[0].keypoints

        new_bboxes.append({
            'transcription': bbox['transcription'], 
            'points': [(
                min(max(0, point.x), image.shape[1] - 1), 
                min(max(0, point.y), image.shape[0] - 1)
            ) for point in keypoints], 
            'difficult': bbox['difficult']
        })
    return image, new_bboxes


def split_regions(axis):
    regions, min_axis_index = [], 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            regions.append(axis[min_axis_index:i])
            min_axis_index = i
    return regions


def random_select(axis):
    xx = np.random.choice(axis, size=2)
    return np.min(xx), np.max(xx)


def region_wise_random_select(regions):
    selected_index = list(np.random.choice(len(regions), 2))
    selected_values = []
    for index in selected_index:
        axis = regions[index]
        selected_values.append(int(np.random.choice(axis, size=1)))
    return min(selected_values), max(selected_values)


def crop(image, image_bboxes, max_tries=10, min_crop_side_ratio=0.1):
    h, w, _ = image.shape
    h_array = np.zeros(h, dtype=np.int32)
    w_array = np.zeros(w, dtype=np.int32)

    for bbox in image_bboxes:
        points = np.round(bbox['points'], decimals=0).astype(np.int32)
        minx, maxx = np.min(points[:, 0]), np.max(points[:, 0])
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1])
        w_array[minx:maxx] = 1
        h_array[miny:maxy] = 1

    # Ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0: return image, image_bboxes
    h_regions = split_regions(h_axis)
    w_regions = split_regions(w_axis)

    for _ in range(max_tries):
        if len(w_regions) > 1: xmin, xmax = region_wise_random_select(w_regions)
        else: xmin, xmax = random_select(w_axis)

        if len(h_regions) > 1: ymin, ymax = region_wise_random_select(h_regions)
        else: ymin, ymax = random_select(h_axis)

        if xmax - xmin < min_crop_side_ratio * w or \
           ymax - ymin < min_crop_side_ratio * h: continue # area too small

        new_bboxes = []
        for bbox in image_bboxes:
            points = np.array(bbox['points'])
            if not (
                points[:, 0].min() > xmax or 
                points[:, 0].max() < xmin or 
                points[:, 1].min() > ymax or 
                points[:, 1].max() < ymin
            ):
                points[:, 0] -= xmin
                points[:, 0] = np.clip(points[:, 0], 0., (xmax - xmin - 1) * 1.)
                points[:, 1] -= ymin
                points[:, 1] = np.clip(points[:, 1], 0., (ymax - ymin - 1) * 1.)
                new_bboxes.append({
                    'transcription': bbox['transcription'], 
                    'points': points.tolist(), 
                    'difficult': bbox['difficult']
                })
        if len(new_bboxes) > 0: return image[ymin:ymax, xmin:xmax], new_bboxes
    return image, image_bboxes


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