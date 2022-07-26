import os
import csv
import tensorflow as tf
from tqdm import tqdm
from loader import DataImporter


class Evaluator:
    def __init__(self, model, dataset_dir, transcripts_path):
        self.model = model
        self.dataset = DataImporter(dataset_dir, transcripts_path, min_length=1)
        self.tf_dataset = None
        
        
    def evaluate(self, data_handler, batch_size, drop_remainder=False):
        self.tf_dataset = tf.data.Dataset.from_tensor_slices((self.dataset.img_paths, self.dataset.labels))
        self.tf_dataset = self.tf_dataset.map(
            lambda img_path, label: (
                data_handler.process_image(img_path), 
                data_handler.process_label(label)
            ), num_parallel_calls = tf.data.AUTOTUNE
        ).batch(batch_size, drop_remainder=drop_remainder)
        
        self.data_handler = data_handler
        self.tf_dataset = self.tf_dataset.cache().prefetch(tf.data.AUTOTUNE)  
        return self.model.evaluate(self.tf_dataset, return_dict=True)
    
    
    def write_csv(self, file_name, use_ctc_decode=False):
        assert self.tf_dataset, "evaluate() method need to be run first"
        with open(file_name, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['img_paths', 'labels', 'pred_labels'])
            
            for idx, (batch_images, batch_tokens) in tqdm(enumerate(self.tf_dataset)):
                labels = self.data_handler.tokens2texts(batch_tokens)
                pred_tokens = self.model.predict(batch_images)
                pred_labels = self.data_handler.tokens2texts(pred_tokens, use_ctc_decode)
                
                batch_size = len(batch_images)
                paths = self.dataset.img_paths[idx * batch_size: (idx + 1) * batch_size]
                paths = ['/'.join(os.path.abspath(path).split(os.path.sep)[-2:]) for path in paths]
                writer.writerows(list(map(list, zip(*[paths, labels, pred_labels]))))