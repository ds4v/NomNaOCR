from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import cv2
import os

ap = ArgumentParser()
ap.add_argument('-i', '--input_dir', required=True, help='Path to images directory')
ap.add_argument('-o', '--output_dir', required=False, help='Images directory after rotation')
args = vars(ap.parse_args())

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, args['input_dir']).strip('/')
output_dir = os.path.join(script_dir, args['output_dir'] or input_dir + ' - Preprocess')


def read_unicode_image_path(image_path):
    stream = bytearray(open(image_path, 'rb').read())
    stream = np.asarray(stream, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)


def preprocess_image(image_path):
    # https://www.kaggle.com/pr1c3f1eld/data-cleaning-and-pre-processing
    image = read_unicode_image_path(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)

    image = read_unicode_image_path(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_image = cv2.add(image, mask)

    blur_image = cv2.GaussianBlur(new_image, (13, 13), 0)
    sharp_mask = np.subtract(new_image, blur_image)
    sharp_mask = cv2.GaussianBlur(sharp_mask, (3, 3), 0)

    new_image = cv2.addWeighted(new_image, 4, sharp_mask, -4, 0)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    return new_image


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print('Created folder:', output_dir)

for file_name in tqdm(os.listdir(input_dir)):
    if not file_name.endswith(('.jpg', '.png', 'jpeg')): continue
    image_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    new_image = preprocess_image(image_path)
    is_success, image_buffer = cv2.imencode('.jpg', new_image)
    if is_success: image_buffer.tofile(output_path)