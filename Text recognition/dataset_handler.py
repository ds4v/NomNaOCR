import os
import re
import numpy as np
import tensorflow as tf
from string import punctuation
from collections import defaultdict, Counter


def is_clean_text(text):
    not_nom_chars = r'\sA-Za-z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ'
    pattern = re.compile(f'[{not_nom_chars}{re.escape(punctuation)}]')
    return not bool(re.search(pattern, text))


def create_dataset(dataset_dir, labels_path, min_length=4):
    img_paths, labels = [], []
    vocabs = defaultdict(int)

    with open(labels_path, 'r', encoding='utf-8') as file:
        for line in file:
            img_name, text = line.rstrip('\n').split('\t')
            img_path = os.path.join(dataset_dir, img_name)

            text = text.strip().lower()
            if os.path.getsize(img_path) and len(text) >= min_length and is_clean_text(text):
                img_paths.append(img_path)
                labels.append(text)
            
    sorted_vocabs = dict(Counter(''.join(labels)).most_common())
    return np.array(img_paths), np.array(labels), sorted_vocabs


def remove_rare_chars(img_paths, labels, sorted_vocabs, threshold=1):
    if threshold < 2: return img_paths, labels, sorted_vocabs
    rare_chars, idxs_to_remove = [], []
    is_satisfy_threshold = True

    # Vocabs need to be sorted (for faster checking)
    for char, freq in reversed(sorted_vocabs.items()):
        if freq < threshold: rare_chars.append(char)
        else: break

    for idx, label in enumerate(labels):
        if any((char in label) for char in rare_chars):
            idxs_to_remove.append(idx)

    # Remove sentences containing rare characters and recalculate the vocab frequencies
    idxs_to_remove = np.array(idxs_to_remove)
    img_paths = np.delete(img_paths, idxs_to_remove)
    labels = np.delete(labels, idxs_to_remove)
    new_vocabs = dict(Counter(''.join(labels)).most_common())

    # Check if there are still rare characters after removing sentences
    smallest_freq = threshold + 1 # If vocabs is empty, the smallest frequency always > threshold
    if len(new_vocabs) >= 1: smallest_freq = list(new_vocabs.values())[-1] 
    no_more_rares = False if smallest_freq < threshold else True

    assert len(img_paths) == len(labels), 'img_paths and labels not same size'
    if no_more_rares: return img_paths, labels, new_vocabs
    return remove_rare_chars(img_paths, labels, new_vocabs, threshold)


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


def process_image(img_path, img_size):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def process_label(label, max_length, char2num_func, extra_tokens={}):
    label = char2num_func(tf.strings.unicode_split(label, input_encoding='UTF-8'))
    if {'START', 'END'} <= extra_tokens.keys(): 
        start_token = extra_tokens['START']
        end_token = extra_tokens['END']
        label = tf.concat([[start_token], label, [end_token]], 0)

    label_length = tf.shape(label, tf.int64)[0]
    label = tf.pad(
        label, 
        paddings = [[0, max_length - label_length]], 
        constant_values = extra_tokens['PAD']
    )
    return label


def prepare_tf_dataset(
    elements, img_size, max_length, batch_size, char2num_func,
    extra_tokens = {'START': None, 'END': None, 'PAD': None},
    drop_remainder = False
):
    dataset = tf.data.Dataset.from_tensor_slices(elements).map(
        lambda img_path, label: (
            process_image(img_path, img_size),
            process_label(label, max_length, char2num_func, extra_tokens)
        ),
        num_parallel_calls = tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size, drop_remainder).cache().prefetch(tf.data.AUTOTUNE)


def tokens2texts(batch_tokens, padding_token, num2char_func, removed_strs=[]):
    batch_texts = []
    for tokens in batch_tokens:
        # Gather indices where label != padding_token.
        not_padding = tf.math.not_equal(tokens, padding_token)
        indices = tf.gather(tokens, tf.where(not_padding))

        # Convert to string
        text = tf.strings.reduce_join(num2char_func(indices)) 
        text = text.numpy().decode('utf-8')

        for string in removed_strs: 
            text = text.replace(string, '')
        batch_texts.append(text)
    return batch_texts
