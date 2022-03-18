from argparse import ArgumentParser
from bbox_handler import BoundingBoxHandler
import json
import sys
import os

ap = ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='File that contains PPOCR rotated bboxes')
ap.add_argument('-o', '--output', required=True, help='File name after combine rotated bboxes')
ap.add_argument(
    '-d',
    '--direction',
    required = True,
    choices = ['+90', '-90', 'both'],
    help = 'Current right angle direction of input images'
)
ap.add_argument(
    '--max_woh',
    required = 'both' in sys.argv[-1],
    type = float,
    help = '(Required if direction == "both") Maximum ratio width over height to filter'
)
ap.add_argument(
    '--overlap',
    required = 'both' in sys.argv[-1],
    type = float,
    help = '(Required if direction == "both") Overlap threshold to suppress'
)
ap.add_argument(
    '--nms_inverse', # Sometimes the non-maximum boxes fit the sentence better
    required = 'both' in sys.argv[-1],
    type = lambda val: val.lower() in ('true', 't', '1'),
    help = '(Required if direction == "both") Inverve indices after NMS or not'
)
args = vars(ap.parse_args())

'''Example:
python unrotated_convertor.py \
    -i "../../Dataset/Tale of Kieu version 1872 - Rotate/Cache.cach" \
    -o "../../Dataset/Tale of Kieu version 1872/Cache.cach" \
    -d "both" \
    --max_woh 0.25 \
    --overlap 0.7 \
    --nms_inverse 1
'''

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, args['input'])
output_path = os.path.join(script_dir, args['output'])


def rotate_bboxes_to_0deg(image_idx, file_path, bboxes):
    angle = int(os.path.splitext(file_path)[0][-3:])
    if args['direction'] == 'both':
        if (image_idx % 2 == 0 and angle != 90) or \
            (image_idx % 2 == 1 and angle != -90): 
            raise Exception('''
                \nImage must have the following format:
                \n- "+90" postfix in name for even index
                \n- "-90" postfix in name for odd index
            ''')
    elif int(args['direction']) != angle:
        raise Exception('Image not meet current right angle direction')

    for idx, bbox in enumerate(bboxes):
        absolute_path = os.path.join(
            os.path.dirname(input_path),
            os.path.basename(file_path)  # Get file name
        )
        bboxes[idx] = BoundingBoxHandler.RotateOneBox(absolute_path, bbox, -angle)
        bboxes[idx]['points'] = BoundingBoxHandler.RectangleTransform(bboxes[idx]['points'])

    print('Rotated', file_path, 'bouding boxes to 0 degree')
    return bboxes


with open(input_path, 'r', encoding='utf-8') as file:
    dataset_bboxes = {}
    for line in file:
        file_path, bboxes = line.rstrip('\n').split('\t')
        dataset_bboxes[file_path] = json.loads(bboxes)

with open(output_path, 'w', encoding='utf-8') as file:
    if args['direction'] in ['+90', '-90']:
        for image_idx, item in enumerate(dataset_bboxes.items()):
            file_path, bboxes = item
            final_path = file_path.replace(args['direction'], '').replace(' - Rotate', '')
            bboxes = rotate_bboxes_to_0deg(image_idx, file_path, bboxes)
            bboxes = BoundingBoxHandler.WidthOverHeightFilter(bboxes, max_ratio=args['max_woh'])
            file.write(f'{final_path}\t{bboxes}\n')

    elif args['direction'] == 'both':
        dataset_length = len(dataset_bboxes)
        if dataset_length % 2 != 0:
            raise Exception('Number of images to rotate must be even')

        items = list(dataset_bboxes.items())
        for image_idx in range(0, dataset_length, 2):
            file_path_1, bboxes_1 = items[image_idx]  # for +90 degree
            file_path_2, bboxes_2 = items[image_idx + 1]  # for -90 degree
            final_path = file_path_1.replace('+90', '').replace(' - Rotate', '')

            bboxes_1 = rotate_bboxes_to_0deg(image_idx, file_path_1, bboxes_1)
            bboxes_2 = rotate_bboxes_to_0deg(image_idx + 1, file_path_2, bboxes_2)
            final_bboxes = BoundingBoxHandler.WidthOverHeightFilter(
                bboxes_1 + bboxes_2,
                max_ratio = args['max_woh']
            )
            final_bboxes = BoundingBoxHandler.NonMaximumSuppression(
                final_bboxes,
                threshold = args['overlap'],
                inverse_idxs = args['nms_inverse']
            )
            print('=> Merged', 'rotated bouding boxes for', final_path)
            file.write(f'{final_path}\t{final_bboxes}\n')
