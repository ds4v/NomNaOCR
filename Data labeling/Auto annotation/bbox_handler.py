from scipy.spatial import distance
from functools import reduce
import numpy as np
import operator
import math
import cv2


class BoundingBoxHandler:
    # https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates
    @staticmethod
    def BlhsingOrderPoints(points):
        center = tuple(map(
            operator.truediv,
            reduce(lambda x, y: map(operator.add, x, y), points),
            [len(points)] * 2
        ))
        return sorted(points, key=lambda point: (
            -135 - math.degrees(math.atan2(*tuple(
                map(operator.sub, point, center)
            )))) % 360)

    # https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv
    @staticmethod
    def AdrianOrderPoints(points):
        # Sort the points based on their x-coordinates
        xSorted = points[np.argsort(points[:, 0]), :]
        # Grab the left-most and right-most points from the sorted x-roodinate points
        leftMost, rightMost = xSorted[:2, :], xSorted[2:, :]
        # Now, sort the left-most coordinates according to their y-coordinates 
        # so we can grab the top-left and bottom-left points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        # Use the top-left coordinate as an anchor to calculate the Euclidean distance
        # between the top-left and right-most points; by the Pythagorean theorem, 
        # the point with the largest distance will be our bottom-right point
        D = distance.cdist(tl[np.newaxis], rightMost, 'euclidean')[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        # Return the coordinates in top-left, top-right, bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype='float32')

    # https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example
    @staticmethod
    def RectangleTransform(points):
        quadrangle = BoundingBoxHandler.AdrianOrderPoints(np.array(points))
        (tl, tr, br, bl) = quadrangle
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB)) + tl[0] - 1
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB)) + tl[1] - 1
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left order
        dst = np.array([
            [tl[0], tl[1]], 
            [maxWidth, tl[1]], 
            [maxWidth, maxHeight], 
            [tl[0], maxHeight]
        ], dtype='float32')
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(quadrangle, dst)
        transformed = cv2.perspectiveTransform(np.array([quadrangle]), M)
        return transformed[0].tolist()

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
    def NonMaximumSuppression(bboxes, threshold):
        if len(bboxes) == 0: return []
        points_in_bboxes = np.array([bbox['points'] for bbox in bboxes])
        pick_idxs = []  # Initialize the list of picked indexes
        pick_bboxes = [] 

        x1 = points_in_bboxes[:, 0, 0] # x coordinate of the top-left corner
        y1 = points_in_bboxes[:, 0, 1] # y coordinate of the top-left corner
        x2 = points_in_bboxes[:, 2, 0] # x coordinate of the bottom-right corner
        y2 = points_in_bboxes[:, 2, 1] # y coordinate of the bottom-right corner

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

        # Return only the bounding boxes that 
        # were picked using the integer data type
        return pick_bboxes

    @staticmethod
    def WidthOverHeightFilter(bboxes, max_ratio=0.5): 
        new_bboxes = []
        for bbox in bboxes:
            (tl, tr, br, bl) = bbox['points']
            w = max(0, br[0] - tl[0] + 1)
            h = max(0, br[1] - tl[1] + 1)
            if w / h < max_ratio: new_bboxes.append(bbox)
        return new_bboxes