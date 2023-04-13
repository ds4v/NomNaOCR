from argparse import ArgumentParser, FileType
from tqdm import tqdm
import json
import os
import re

ap = ArgumentParser()
ap.add_argument(
    '-i', 
    '--input_file',
    required = True,
    help = 'PaddleOCR Label.txt path',
)
ap.add_argument(
    '-o', 
    '--output_dir', 
    required = True, 
    help = 'Output directory for IC15 *.txt files'
)

args = vars(ap.parse_args())
if not os.path.exists(args['output_dir']):
    os.makedirs(args['output_dir'])

with open(args['input_file'], 'r', encoding='utf-8') as paddle_txt:
    for line in tqdm(paddle_txt):
        page_path, page_boxes = line.strip().split('\t')
        page_name = page_path.replace('imgs/', '').split('.jpg')[0]
        page_boxes = json.loads(page_boxes)

        with open(f"{args['output_dir']}/{page_name}.txt", 'w', encoding='utf-8') as ic15_txt:
            for box in page_boxes:
                flatten_box = sum(box['points'], [])
                ic15_box = ','.join([str(num * 1.0) for num in flatten_box])
                ic15_txt.write(f'{ic15_box},{box["transcription"]}\n')
