# https://www.tensorflow.org/text/tutorials/nmt_with_attention
# https://www.tensorflow.org/tutorials/text/image_captioning
import tensorflow as tf
from tensorflow.keras.models import clone_model
from abc import abstractmethod, ABCMeta # For define pure virtual functions
from utils import update_tensor_column


def get_imagenet_model(model_name, input_shape):
    # Pick a model from https://keras.io/api/applications
    base_model = eval('tf.keras.applications.' + model_name)
    return base_model(input_shape=input_shape, weights=None, include_top=False)


class CustomTrainingModel(tf.keras.Model, metaclass=ABCMeta):
    def __init__(self, data_handler=None, name='CustomTrainingModel', **kwargs):
        super(CustomTrainingModel, self).__init__(name=name, **kwargs)
        self.data_handler = data_handler

    @abstractmethod
    def get_config(self):
        pass # Pure virtual functions => Must be overridden in the derived classes

    @classmethod
    def from_config(cls, config):
        return cls(**config) # To clone model when using kfold training

    @tf.function
    def _update_metrics(self, batch):
        batch_images, batch_tokens = batch
        predictions = self.predict(batch_images) 
        self.compiled_metrics.update_state(batch_tokens, predictions)
        return {m.name: m.result() for m in self.metrics}

    @abstractmethod
    def _compute_loss_and_metrics(self):
        pass # Pure virtual functions => Must be overridden in the derived classes

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            loss, display_results = self._compute_loss_and_metrics(batch)

        # Apply an optimization step
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return display_results

    @tf.function
    def test_step(self, batch):
        _, display_results = self._compute_loss_and_metrics(batch)
        return display_results
    
    @tf.function
    def _init_pred_tokens(self, batch_size, return_new_tokens=True):
        seq_tokens = tf.fill([batch_size, self.data_handler.max_length], 0)
        seq_tokens = tf.cast(seq_tokens, dtype=tf.int64)
        new_tokens = tf.fill([batch_size, 1], self.data_handler.start_token)
        new_tokens = tf.cast(new_tokens, dtype=tf.int64)

        seq_tokens = update_tensor_column(seq_tokens, new_tokens, 0)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        if not return_new_tokens: return seq_tokens, done
        return seq_tokens, new_tokens, done

    @tf.function
    def _update_pred_tokens(self, y_pred, seq_tokens, done, pos_idx, return_new_tokens=True):
        # Set the logits for all masked tokens to -inf, so they are never chosen
        y_pred = tf.where(self.data_handler.token_mask, float('-inf'), y_pred)
        new_tokens = tf.argmax(y_pred, axis=-1) 

        # Add batch dimension if it is not present after argmax
        if tf.rank(new_tokens) == 1: new_tokens = tf.expand_dims(new_tokens, axis=1)

        # Once a sequence is done it only produces padding token
        new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)
        seq_tokens = update_tensor_column(seq_tokens, new_tokens, pos_idx)

        # If a sequence produces an `END_TOKEN`, set it `done` after that
        done = done | (new_tokens == self.data_handler.end_token)
        if not return_new_tokens: return seq_tokens, done
        return seq_tokens, new_tokens, done

    @abstractmethod
    def predict(self, batch_images):
        pass # Pure virtual functions => Must be overridden in the derived classes


class EncoderDecoderModel(CustomTrainingModel):
    def __init__(
        self, 
        encoder: tf.keras.Model,
        decoder: tf.keras.Model, 
        data_handler = None, # DataHandler instance
        dec_rnn_name = '', # Use if there is no rnn in the encoder
        name = 'EncoderDecoderModel', 
        **kwargs
    ):
        super(EncoderDecoderModel, self).__init__(data_handler, name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.dec_rnn_name = dec_rnn_name


    def get_config(self):
        return{
            'encoder': clone_model(self.encoder), 
            'decoder': clone_model(self.decoder),
            'dec_rnn_name': self.dec_rnn_name,
            'data_handler': self.data_handler
        }
    

    @tf.function
    def _compute_loss_and_metrics(self, batch):
        batch_images, batch_tokens = batch
        batch_size = batch_images.shape[0]
        loss = tf.constant(0.0)
        
        dec_input = tf.expand_dims([self.data_handler.start_token] * batch_size, 1) 
        if self.dec_rnn_name: 
            enc_output = self.encoder(batch_images)
            dec_units = self.decoder.get_layer(self.dec_rnn_name).units
            hidden = tf.zeros((batch_size, dec_units), dtype=tf.float32)
        else: enc_output, hidden = self.encoder(batch_images)
            
        for i in range(1, self.data_handler.max_length):
            # Passing the features through the decoder
            y_pred, hidden, _ = self.decoder([dec_input, enc_output, hidden])
            loss += self.loss(batch_tokens[:, i], y_pred) 
            
            # Use teacher forcing
            dec_input = tf.expand_dims(batch_tokens[:, i], 1) 
        
        # Update training display result
        metrics = self._update_metrics(batch)
        return loss, {'loss': loss / self.data_handler.max_length, **metrics}
    

    @tf.function
    def predict(self, batch_images, return_attention=False):
        batch_size = batch_images.shape[0]
        seq_tokens, new_tokens, done = self._init_pred_tokens(batch_size)
        attentions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        if self.dec_rnn_name: 
            enc_output = self.encoder(batch_images)
            dec_units = self.decoder.get_layer(self.dec_rnn_name).units
            hidden = tf.zeros((batch_size, dec_units), dtype=tf.float32)
        else: enc_output, hidden = self.encoder(batch_images)

        for i in range(1, self.data_handler.max_length):
            y_pred, hidden, attention_weights = self.decoder([new_tokens, enc_output, hidden])
            attentions = attentions.write(i - 1, attention_weights)
            seq_tokens, new_tokens, done = self._update_pred_tokens(y_pred, seq_tokens, done, i)
            if tf.executing_eagerly() and tf.reduce_all(done): break

        if not return_attention: return seq_tokens
        return seq_tokens, tf.transpose(tf.squeeze(attentions.stack()), [1, 0, 2])


class PartialImageCaptioner(CustomTrainingModel):
    def __init__(
        self, 
        captioner: tf.keras.Model, 
        data_handler = None, # DataHandler instance
        name = 'PartialImageCaptioner', 
        **kwargs
    ):
        super(PartialImageCaptioner, self).__init__(data_handler, name, **kwargs)
        self.captioner = captioner


    def get_config(self):
        return {
            'captioner': clone_model(self.captioner), 
            'data_handler': self.data_handler,
        }


    @tf.function
    def _compute_loss_and_metrics(self, batch):
        batch_images = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        batch_partial_tokens = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True) 
        batch_target_tokens = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        count = 0

        ''' Process the input into this format:
            Image vector + A -> girl
            Image vector + A girl -> going
            Image vector + A girl going -> into
            Image vector + A girl going into -> a
            Image vector + A girl going into a -> wooden
            Image vector + A girl going into a wooden -> building
        Example: https://images.viblo.asia/7f1b412d-9f4e-4b36-aa51-63ce958f84e2.jpeg
        '''
        for item_idx in range(batch[0].shape[0]): # Loop through each item in batch
            image, seq_tokens = batch[0][item_idx], batch[1][item_idx]
            label_length = tf.math.count_nonzero(seq_tokens, dtype=tf.int32)
            
            for token_idx in range(1, tf.math.count_nonzero(seq_tokens, dtype=tf.int32)):
                partial_tokens = tf.pad(
                    seq_tokens[:token_idx], 
                    paddings = [[0, self.data_handler.max_length - token_idx - 1]], 
                    constant_values = 0 # Pad with padding token
                )
                batch_images = batch_images.write(count, image)
                batch_partial_tokens = batch_partial_tokens.write(count, partial_tokens)
                batch_target_tokens = batch_target_tokens.write(count, seq_tokens[token_idx])
                count += 1

        # Get predictions and compute the loss
        y_pred = self.captioner([batch_images.stack(), batch_partial_tokens.stack()])
        loss = self.loss(batch_target_tokens.stack(), y_pred) 

        # Update training display result
        # metrics = self._update_metrics(batch)
        return loss, {'loss': loss}

    
    @tf.function
    def predict(self, batch_images):
        pass