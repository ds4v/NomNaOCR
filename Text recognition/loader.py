import os
import re
import numpy as np
import tensorflow as tf
from string import punctuation
from collections import defaultdict, Counter


class DataImporter:
    def __init__(self, dataset_dir, labels_path, min_length=4):
        self.img_paths, self.labels = [], []
        with open(labels_path, 'r', encoding='utf-8') as file:
            for line in file:
                img_name, text = line.rstrip('\n').split('\t')
                img_path = os.path.join(dataset_dir, img_name)
                text = text.strip().lower()

                if os.path.getsize(img_path) and len(text) >= min_length and self.is_clean_text(text):
                    self.img_paths.append(img_path)
                    self.labels.append(text)
                
        self.img_paths = np.array(self.img_paths)
        self.labels = np.array(self.labels)
        self.vocabs = dict(Counter(''.join(self.labels)).most_common())


    def remove_rare_chars(self, threshold=1):
        if threshold < 2: return self
        rare_chars, idxs_to_remove = [], []
        is_satisfy_threshold = True

        # Vocabs need to be sorted (for faster checking)
        for char, freq in reversed(self.vocabs.items()):
            if freq < threshold: rare_chars.append(char)
            else: break

        for idx, label in enumerate(self.labels):
            if any((char in label) for char in rare_chars):
                idxs_to_remove.append(idx)

        # Remove sentences containing rare characters and recalculate the vocab frequencies
        idxs_to_remove = np.array(idxs_to_remove)
        self.img_paths = np.delete(self.img_paths, idxs_to_remove)
        self.labels = np.delete(self.labels, idxs_to_remove)
        self.vocabs = dict(Counter(''.join(self.labels)).most_common())

        # Check if there are still rare characters after removing sentences
        smallest_freq = threshold + 1 # If vocabs is empty, the smallest frequency always > threshold
        if len(self.vocabs) >= 1: smallest_freq = list(self.vocabs.values())[-1] 
        no_more_rares = False if smallest_freq < threshold else True

        assert len(self.img_paths) == len(self.labels), 'img_paths and labels not same size'
        if no_more_rares: return self
        return self.remove_rare_chars(threshold)


    def is_clean_text(self, text):
        not_nom_chars = r'\sA-Za-z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ'
        pattern = re.compile(f'[{not_nom_chars}{re.escape(punctuation)}]')
        return not bool(re.search(pattern, text))


    def __str__(self):
        return (
            f'Number of images found: {len(self.img_paths)}\n'
            f'Number of labels found: {len(self.labels)}\n'
            f'Number of unique characters: {len(self.vocabs)}\n'
            f'Characters present: {self.vocabs}'
        )


class DataInputPipeline:
    def __init__(self, dataset: DataImporter, img_size: tuple, padding_char, start_char='', end_char=''):
        self.img_paths = dataset.img_paths
        self.labels = dataset.labels
        self.vocabs = dataset.vocabs

        self.img_size = img_size
        self.padding_char = padding_char
        self.start_char = start_char
        self.end_char = end_char

        # Mapping characters to integers
        vocabulary = list(self.vocabs)
        if start_char != '' and end_char != '': vocabulary += [start_char, end_char]
        self.char2num = tf.keras.layers.StringLookup(vocabulary=vocabulary, mask_token=padding_char)

        # Mapping integers back to original characters
        self.num2char = tf.keras.layers.StringLookup(
            vocabulary = self.char2num.get_vocabulary(), 
            mask_token = padding_char, 
            invert = True,
        )

        self.max_length = max([len(label) for label in self.labels])
        self.padding_token = self.char2num(padding_char)
        self.start_token, self.end_token = None, None
        self.start_concat, self.end_concat = [], []

        if self.start_char != '' and self.end_char != '': 
            self.start_token = self.char2num(start_char)
            self.end_token = self.char2num(end_char)
            self.start_concat = [self.start_token]
            self.end_concat = [self.end_token]
            self.max_length += 2 # For [START] and [END] tokens


    def distortion_free_resize(self, image):
        h, w = self.img_size
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


    def process_image(self, img_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, 1)
        image = self.distortion_free_resize(image)
        image = tf.cast(image, tf.float32) / 255.0
        return image


    def process_label(self, label):
        label = self.char2num(tf.strings.unicode_split(label, input_encoding='UTF-8'))
        label = tf.concat([self.start_concat, label, self.end_concat], 0)
        label_length = tf.shape(label, tf.int64)[0]
        label = tf.pad(
            label, 
            paddings = [[0, self.max_length - label_length]], 
            constant_values = self.padding_token
        )
        return label


    def prepare_tf_dataset(self, idxs, batch_size, drop_remainder=False):
        dataset = tf.data.Dataset.from_tensor_slices((self.img_paths[idxs], self.labels[idxs])).map(
            lambda img_path, label: (self.process_image(img_path), self.process_label(label)),
            num_parallel_calls = tf.data.AUTOTUNE
        )
        return dataset.batch(batch_size, drop_remainder).cache().prefetch(tf.data.AUTOTUNE)  


    def tokens2texts(self, batch_tokens):
        batch_texts = []
        for tokens in batch_tokens:
            # Gather indices where label != padding_token.
            not_padding = tf.math.not_equal(tokens, self.padding_token)
            indices = tf.gather(tokens, tf.where(not_padding))

            # Convert to string
            text = tf.strings.reduce_join(self.num2char(indices)) 
            text = text.numpy().decode('utf-8')
            text = text.replace(self.start_char, '').replace(self.end_char, '')
            batch_texts.append(text)
        return batch_texts  
