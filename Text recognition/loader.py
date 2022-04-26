import os
import re
import numpy as np
import tensorflow as tf

from utils import ctc_decode
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
        
        assert len(self.img_paths) == len(self.labels), 'img_paths and labels must have same size'
        self.img_paths = np.array(self.img_paths)
        self.labels = np.array(self.labels)
        self.vocabs = dict(Counter(''.join(self.labels)).most_common())
        self.size = len(self.labels)


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

        assert len(self.img_paths) == len(self.labels), 'img_paths and labels must have same size'
        self.vocabs = dict(Counter(''.join(self.labels)).most_common())
        self.size = len(self.labels)

        # Check if there are still rare characters after removing sentences
        smallest_freq = threshold + 1 # If vocabs is empty, the smallest frequency always > threshold
        if len(self.vocabs) >= 1: smallest_freq = list(self.vocabs.values())[-1] 
        return self.remove_rare_chars(threshold) if smallest_freq < threshold else self


    def is_clean_text(self, text):
        not_nom_chars = r'\sA-Za-z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ'
        pattern = re.compile(f'[{not_nom_chars}{re.escape(punctuation)}]')
        return not bool(re.search(pattern, text))


    def __str__(self):
        return (
            f'Samples count (not include Latin letters, numbers, punctuations):'
            f'\n- Number of images found: {len(self.img_paths)}'
            f'\n- Number of labels found: {len(self.labels)}'
            f'\n- Number of unique characters: {len(self.vocabs)}'
            f'\n- Characters present: {self.vocabs}'
        )


class DataHandler:
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
        self.start_token, self.end_token = None, None
        self.start_concat, self.end_concat = [], []
        mask_idxs = [0, 1] # For [PAD] and [UNK] tokens

        if self.start_char != '' and self.end_char != '': 
            self.start_token = self.char2num(start_char)
            self.end_token = self.char2num(end_char)
            self.start_concat = [self.start_token]
            self.end_concat = [self.end_token]
            self.max_length += 2 # For [START] and [END] tokens
            mask_idxs.append(self.start_token)

        # Prevent from generating padding, unknown, or start when using argmax in model.predict
        token_mask = np.zeros([self.char2num.vocab_size()], dtype=bool)
        token_mask[np.array(mask_idxs)] = True
        self.token_mask = token_mask


    def distortion_free_resize(self, image, align_top=True):
        h, w = self.img_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        # Check the amount of padding needed to be done.
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]
        if pad_height == 0 and pad_width == 0: return image

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
            [0, pad_height_top + pad_height_bottom] if align_top else [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ], constant_values=255) # Pad with white color


    def process_image(self, img_path, img_align_top=True):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, 3)
        image = self.distortion_free_resize(image, img_align_top)
        image = tf.cast(image, tf.float32) / 255.0
        return image


    def process_label(self, label):
        label = self.char2num(tf.strings.unicode_split(label, input_encoding='UTF-8'))
        label = tf.concat([self.start_concat, label, self.end_concat], 0)
        label_length = tf.shape(label, tf.int64)[0]
        label = tf.pad(
            label, 
            paddings = [[0, self.max_length - label_length]], 
            constant_values = 0 # Pad with padding token
        )
        return label


    def prepare_tf_dataset(self, idxs, batch_size, drop_remainder=False, img_align_top=True, use_cache=True):
        dataset = tf.data.Dataset.from_tensor_slices((self.img_paths[idxs], self.labels[idxs])).map(
            lambda img_path, label: (
                self.process_image(img_path, img_align_top), 
                self.process_label(label)
            ), num_parallel_calls = tf.data.AUTOTUNE
        ).batch(batch_size, drop_remainder=drop_remainder)

        # When use .cache(), everything before is saved in the memory. It gives a 
        # significant boost in speed but only if you can get your hands on a larger RAM
        if use_cache: dataset = dataset.cache()
        return dataset.prefetch(tf.data.AUTOTUNE)  


    def tokens2texts(self, batch_tokens, use_ctc_decode=False):
        batch_texts = []
        if use_ctc_decode: 
            batch_tokens = ctc_decode(batch_tokens, self.max_length)

        # Iterate over the results and get back the text
        for tokens in batch_tokens:
            indices = tf.gather(tokens, tf.where(tf.logical_and(
                tokens != 0, # For [PAD] token
                tokens != -1 # For blank label if use_ctc_decode 
            )))

            # Convert to string
            text = tf.strings.reduce_join(self.num2char(indices)) 
            text = text.numpy().decode('utf-8')
            text = text.replace(self.start_char, '').replace(self.end_char, '')
            batch_texts.append(text)
        return batch_texts 
