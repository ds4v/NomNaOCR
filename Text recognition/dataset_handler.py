import os
import re
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict


def create_dataset(dataset_path):
    raw_paths = list(map(str, Path(dataset_path).glob('*.jpg')))
    img_paths, labels = [], []
    vocabs = defaultdict(int)

    for path in raw_paths:
        if os.path.getsize(path):
            img_paths.append(path)
            label = re.sub('_.*', '', os.path.basename(path))
            labels.append(label)
            for char in label: vocabs[char] += 1

    vocabs = dict(sorted(
        vocabs.items(),
        key = lambda item: item[1],
        reverse = True
    ))
    return np.array(img_paths), np.array(labels), vocabs


def remove_rare_chars(img_paths, labels, vocabs, threshold=3):
    rare_chars, new_vocabs = [], vocabs.copy()
    for char, freq in vocabs.items():
        if freq < threshold:
            rare_chars.append(char)
            del new_vocabs[char]

    idxs_to_remove = []
    for idx, label in enumerate(labels):
        if any((char in label) for char in rare_chars):
            idxs_to_remove.append(idx)

    idxs_to_remove = np.array(idxs_to_remove)
    img_paths = np.delete(img_paths, idxs_to_remove)
    labels = np.delete(labels, idxs_to_remove)

    assert len(img_paths) == len(labels), 'img_paths and labels not same size'
    return img_paths, labels, new_vocabs


def distortion_free_resize(image, img_size):
    h, w = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check the amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top, pad_height_bottom = height + 1, height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left, pad_width_right = width + 1, width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    return tf.pad(image, paddings=[
        [pad_height_top, pad_height_bottom],
        [pad_width_left, pad_width_right],
        [0, 0],
    ], constant_values=255)  # Pad with white color


def process_one_image(image_path, img_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image
