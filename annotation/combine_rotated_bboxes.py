from argparse import ArgumentParser
from bbox_handler import BoundingBoxHandler
import json
import os

ap = ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='file that contains PPOCR rotated bboxes')
ap.add_argument('-o', '--output', required=True, help='file name after combine rotated bboxes')
args = vars(ap.parse_args())
'''Example:
python combine_rotated_bboxes.py \
    -i "../Dataset/Tale of Kieu version 1866 - Rotate/Cache.cach" \
    -o "../Dataset/Tale of Kieu version 1866/Cache.cach" \
'''

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, args['input'])
output_path = os.path.join(script_dir, args['output'])


def rotate_bboxes_to_0deg(image_idx, file_path, bboxes):
    angle = int(os.path.splitext(file_path)[0][-3:])
    if (image_idx % 2 == 0 and angle != 90) or (image_idx % 2 == 1 and angle != -90):
        raise Exception('''
            \nImage must have the following format:
            \n- "+90" postfix in name for even index
            \n- "-90" postfix in name for odd index
        ''')

    for idx, bbox in enumerate(bboxes):
        absolute_path = os.path.join(
            os.path.dirname(input_path),
            os.path.basename(file_path)  # Get file name
        )
        bboxes[idx] = BoundingBoxHandler.RotateOneBox(absolute_path, bbox, -angle)
        bboxes[idx]['points'] = BoundingBoxHandler.OrderPoints(bboxes[idx]['points'])

    print('Rotated', file_path, 'bouding boxes to 0 degree')
    return bboxes


with open(input_path, 'r', encoding='utf-8') as file:
    dataset_bboxes = {}
    for line in file:
        file_path, bboxes = line.rstrip('\n').split('\t')
        dataset_bboxes[file_path] = json.loads(bboxes)

dataset_length = len(dataset_bboxes)
if dataset_length % 2 != 0:
    raise Exception('Number of images to rotate must be even')

with open(output_path, 'w', encoding='utf-8') as file:
    items = list(dataset_bboxes.items())
    for image_idx in range(0, dataset_length, 2):
        file_path_1, bboxes_1 = items[image_idx]  # for +90 degree
        file_path_2, bboxes_2 = items[image_idx + 1]  # for -90 degree
        final_path = file_path_1.replace('+90', '').replace(' - Rotate', '')

        bboxes_1 = rotate_bboxes_to_0deg(image_idx, file_path_1, bboxes_1)
        bboxes_2 = rotate_bboxes_to_0deg(image_idx + 1, file_path_2, bboxes_2)
        bboxes = BoundingBoxHandler.NonMaxSuppression(bboxes_1 + bboxes_2, 0.8)

        print('=> Merged', 'rotated bouding boxes for', final_path)
        file.write(f'{final_path}\t{bboxes}\n')
