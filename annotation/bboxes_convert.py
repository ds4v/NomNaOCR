from argparse import ArgumentParser
from converter import PPOCRLabelConverter
import os

ap = ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='path to file that contains PPOCR bboxes')
ap.add_argument('-o', '--output', required=True, help='file name after convert bboxes')
ap.add_argument('-m', '--mode', type=int, required=True, help='\
    1: Merge pairs of rotated boxes, \
    2: Convert to Label Studio format \
')
args = vars(ap.parse_args())

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, args['input'])
output_path = os.path.join(script_dir, args['output'])

bboxes_converter = PPOCRLabelConverter(input_path, output_path)
if args['mode'] == 1: bboxes_converter.MergeRotatedBoxes()
elif args['mode'] == 2: bboxes_converter.ToLabelStudioFormat()
