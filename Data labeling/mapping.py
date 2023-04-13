'''
Implementation of label mapping process:
- Author: Nguyen Duc Duy Anh
- GitHub: https://github.com/duyanh1909
'''
import os
import re
import ast
import glob
import numpy as np


ROOT_PATH = ''
MAP_FOLDER_NAME = ''
LIST_DATA = [ 
    'Luc Van Tien', 
    'Tale of Kieu 1866', 
    'Tale of Kieu 1871', 
    'Tale of Kieu 1872', 
    'DVSKTT-1 Quyen thu', 
    'DVSKTT-2 Ngoai ky toan thu', 
    'DVSKTT-3 Ban ky toan thu', 
    'DVSKTT-4 Ban ky thuc luc', 
    'DVSKTT-5 Ban ky tuc bien'
]


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


def split_detail(text):
    url, list_dict = text.split('\t')
    list_dict = ast.literal_eval(
        list_dict
        .replace('\n', '')
        .replace('false', 'False')
        .replace('true', 'True')
    )
    
    total_word = 0
    for idx, elem in enumerate(list_dict):
        transcription = list_dict[idx]['transcription']
        if '-' not in transcription:
            str_length = '8' if int(transcription) % 2 == 0 else '6'
            transcription += '-' + str_length

        pos = int(re.findall('(\d+)-?', transcription)[0])
        count = int(re.findall('\d+-(\d+)', transcription)[0])
        
        list_dict[idx]['pos'] = pos
        list_dict[idx]['count'] = count
        total_word += count

    list_dict = sorted(list_dict, key=lambda d: d['pos'])
    return {
        "url": url,
        "bbox": list_dict,
        "total": total_word,
        "img": os.path.basename(url)
    }
    
    
for path in LIST_DATA:
    dataset = []
    with open(os.path.join(ROOT_PATH, path, 'Label.txt'), 'r') as f:
        dataset = [split_detail(data) for data in f.readlines()]

    if not os.path.exists(os.path.join(ROOT_PATH, path, MAP_FOLDER_NAME)):
        os.makedirs(os.path.join(ROOT_PATH, path, MAP_FOLDER_NAME))

    for data in dataset:
        final_detect_data = []
        for box in data['bbox']:
            text = box['transcription']
            if box['difficult']: text = '###'
            
            points = list(order_points_clockwise(box['points']).flat)
            points = ','.join([str(num) for num in points])
            final_detect_data.append(f"{points},{text}")
        
        saved_path = os.path.join(ROOT_PATH, path, MAP_FOLDER_NAME, data['img'].replace('jpg', 'txt'))
        with open(saved_path, 'w+') as f:
            f.write('\n'.join(final_detect_data))
            

for path in glob.glob(f'{ROOT_PATH}/*/{MAP_FOLDER_NAME}/*.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        if len(f.readlines()) > 35: print(path)
        
''' Output:
ROOT_PATH/DVSKTT-2 Ngoai ky toan thu/MAP_FOLDER_NAME/DVSKTT_ngoai_II_17a.txt
ROOT_PATH/DVSKTT-2 Ngoai ky toan thu/MAP_FOLDER_NAME/DVSKTT_ngoai_V_25b.txt
'''
