from utils import *
import json
import os

class PPOCRLabelConverter:
    def __init__(self, bboxes_file_path, output_path):
        self.bboxes_file_path = bboxes_file_path
        self.output_path = output_path
        self.dataset_bboxes = {}

        with open(bboxes_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                file_path, bboxes = line.rstrip('\n').split('\t')
                self.dataset_bboxes[file_path] = json.loads(bboxes)

    def __RotateBoxesTo0Deg(self, image_idx, file_path, bboxes):
        angle = int(os.path.splitext(file_path)[0][-3:])
        if (image_idx % 2 == 0 and angle != 90) or \
            (image_idx % 2 == 1 and angle != -90):
            raise Exception('''
                \nImage must have the following format:
                \n- "+90" postfix in name for even index
                \n- "-90" postfix in name for odd index
            ''')

        for bbox_idx, bbox in enumerate(bboxes):
            absolute_file_path = os.path.join(
                os.path.dirname(self.bboxes_file_path),
                os.path.basename(file_path)  # Get file name
            )
            bboxes[bbox_idx] = rotate_bbox(absolute_file_path, bbox, -angle)
        print('Rotated', file_path, 'bouding boxes to 0 degree')
        return bboxes

    def MergeRotatedBoxes(self):
        dataset_length = len(self.dataset_bboxes)
        if dataset_length % 2 != 0:
            raise Exception('Number of images to rotate must be even')

        with open(self.output_path, 'w', encoding='utf-8') as file:
            items = list(self.dataset_bboxes.items())
            for image_idx in range(0, dataset_length, 2):
                file_path_1, bboxes_1 = items[image_idx]  # for +90 degree
                file_path_2, bboxes_2 = items[image_idx + 1]  # for -90 degree
                final_path = file_path_1.replace('+90', '').replace(' - Rotate', '')

                bboxes_1 = self.__RotateBoxesTo0Deg(image_idx, file_path_1, bboxes_1)
                bboxes_2 = self.__RotateBoxesTo0Deg(image_idx + 1, file_path_2, bboxes_2)
                bboxes = non_max_suppression(bboxes_1 + bboxes_2, 0.8)

                print('=> Merged', 'rotated bouding boxes for', final_path)
                file.write(f'{final_path}\t{bboxes}\n')

    def ToLabelStudioFormat(self):
        return
