import os
import re
import numpy as np
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


def split_data(img_paths, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(img_paths)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle: np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = img_paths[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = img_paths[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid
