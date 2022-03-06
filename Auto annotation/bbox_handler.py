from functools import reduce
import numpy as np
import operator
import math
import cv2

class BoundingBoxHandler:
    # https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates
    @staticmethod
    def OrderPoints(points):
        center = tuple(map(
            operator.truediv,
            reduce(lambda x, y: map(operator.add, x, y), points),
            [len(points)] * 2
        ))
        return sorted(points, key=lambda point: (
            -135 - math.degrees(math.atan2(*tuple(
                map(operator.sub, point, center)
            )))) % 360)

    # https://cristianpb.github.io/blog/image-rotation-opencv
    @staticmethod
    def RotateOneBox(file_name, bbox, angle):
        image = cv2.imread(file_name)
        height, width = image.shape[:2]
        center_x, center_y = width / 2, height / 2

        for idx, point in enumerate(bbox['points']):
            # OpenCV calculates standard transformation matrix
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            # Grab the rotation components of the matrix)
            cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
            # Compute the new bounding dimensions of the image
            new_width = (height * sin) + (width * cos)
            new_height = (height * cos) + (width * sin)
            # Adjust the rotation matrix to take into account translation
            M[0, 2] += (new_width / 2) - center_x
            M[1, 2] += (new_height / 2) - center_y
            # Perform the actual rotation and return the image
            calculated = M @ [point[0], point[1], 1]
            bbox['points'][idx] = (calculated[0], calculated[1])
        return bbox

    # https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python
    @staticmethod
    def NonMaximumSuppression(bboxes, threshold, inverse_idxs=False):
        if len(bboxes) == 0: return []
        np_bboxes = np.array([bbox['points'] for bbox in bboxes])
        pick_idxs = []  # Initialize the list of picked indexes
        pick_bboxes = [] 

        x1 = np_bboxes[:, 0, 0] # x coordinate of the top-left corner
        y1 = np_bboxes[:, 0, 1] # y coordinate of the top-left corner
        x2 = np_bboxes[:, 2, 0] # x coordinate of the bottom-right corner
        y2 = np_bboxes[:, 2, 1] # y coordinate of the bottom-right corner

        # Compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # Keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # Grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick_idxs.append(i)
            pick_bboxes.append(bboxes[i])

            # Find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # Delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate((
                [last],
                np.where(overlap > threshold)[0]
            )))

        # Return only the bounding boxes that were picked using the integer data type
        if not inverse_idxs: return pick_bboxes
        return [bbox for idx, bbox in enumerate(bboxes) if idx not in pick_idxs]