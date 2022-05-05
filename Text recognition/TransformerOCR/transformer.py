# https://www.tensorflow.org/text/tutorials/transformer
# https://keras.io/examples/nlp/neural_machine_translation_with_transformer
# https://keras.io/examples/vision/image_captioning
# https://medium.com/geekculture/scene-text-recognition-using-resnet-and-transformer-c1f2dd0e69ae
import sys
sys.path.append('..')

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, MultiHeadAttention, LayerNormalization
from models import EncoderDecoderModel


def point_wise_ffn(embedding_dim, feed_forward_units, dropout_rate=None):
    ffn = tf.keras.Sequential() 
    ffn.add(Dense(feed_forward_units, activation='relu')) # (batch_size, seq_length, feed_forward_units)
    if dropout_rate: ffn.add(Dropout(dropout_rate))
    ffn.add(Dense(embedding_dim)) # (batch_size, seq_length, embedding_dim)
    return ffn


class TransformerEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim, # d_model
        seq_length, # For positional embedding => receptive_size if features or max_length - 1 if texts
        vocab_size = None, # If None => Disable tokens embedding, just use positional embedding
        name = 'TransformerEmbedding',
        **kwargs
    ):
        ''' https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11/303
        We'll learn position-embedding vectors the same way we learn to embed word indices.
        We'll then proceed to add our position embeddings to the corresponding word embeddings,
        to obtain a position-aware word embedding. This technique is called "positional embedding".
        '''
        super(TransformerEmbedding, self).__init__(name=name, **kwargs)
        self.tokens_embedding = Embedding(vocab_size, embedding_dim) if vocab_size else None
        self.positions_embedding = Embedding(seq_length, embedding_dim)
        self.embed_scale = tf.sqrt(tf.cast(embedding_dim, tf.float32))
        self.seq_length = seq_length

    def call(self, inputs, use_scale=True):
        positions = tf.range(start=0, limit=self.seq_length, delta=1)
        embedded_positions = self.positions_embedding(positions)
        if self.tokens_embedding: inputs = self.tokens_embedding(inputs)
        if use_scale: inputs *= self.embed_scale
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        # This will be propagated to the mask argument when call the TransformerDecoderLayer below
        if self.tokens_embedding: return inputs != 0 # Mask padding token
        return None # Not use masking if just use positional embedding


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        embedding_dim, # d_model
        feed_forward_units, # dff
        dropout_rate = 0.1,
        name = 'TransformerEncoderLayer',
        **kwargs
    ):
        super(TransformerEncoderLayer, self).__init__(name=name, **kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)
        self.ffn = point_wise_ffn(embedding_dim, feed_forward_units, dropout_rate)

        # The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the
        # embedding_dim (last) axis. (epsilon: small float added to variance to avoid dividing by 0)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.supports_masking = True

    def call(self, features):
        attn_output = self.mha(query=features, value=features, key=features, attention_mask=None)
        out1 = self.layernorm1(features + attn_output) 
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2 # (batch_size, receptive_size, embedding_dim)


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        embedding_dim, # d_model
        feed_forward_units, # dff
        dropout_rate = 0.1,
        name = 'TransformerDecoderLayer',
        **kwargs
    ):
        super(TransformerDecoderLayer, self).__init__(name=name, **kwargs)
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)
        self.ffn = point_wise_ffn(embedding_dim, feed_forward_units, dropout_rate)

        # The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the
        # embedding_dim (last) axis. (epsilon: small float added to variance to avoid dividing by 0)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.supports_masking = True

    def call(self, inputs, enc_output, mask=None):
        # enc_output.shape == (batch_size, receptive_size, embedding_dim)
        padding_mask = None
        if mask is not None: # Check for the attention mask when use encoder output in MultiHeadAttention
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)

        causal_mask = self.get_causal_attention_mask(inputs)
        look_ahead_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
        look_ahead_mask = tf.minimum(look_ahead_mask, causal_mask)

        attn1, attn_weights_block1 = self.mha1(
            query=inputs, value=inputs, key=inputs, 
            attention_mask=look_ahead_mask, return_attention_scores=True
        )  # (batch_size, max_length - 1, embedding_dim)
        out1 = self.layernorm1(attn1 + inputs)

        attn2, attn_weights_block2 = self.mha2(
            query=out1, value=enc_output, key=enc_output, 
            attention_mask=padding_mask, return_attention_scores=True
        )  # (batch_size, max_length - 1, embedding_dim)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2) # (batch_size, max_length - 1, embedding_dim)
        out3 = self.layernorm3(ffn_output + out2) # (batch_size, max_length - 1, embedding_dim)
        return out3, attn_weights_block1, attn_weights_block2

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_length = input_shape[0], input_shape[1]
        seq_range = tf.range(seq_length)
        
        mask = tf.cast(seq_range[:, tf.newaxis] >= seq_range, dtype=tf.int32)
        mask = tf.reshape(mask, (1, seq_length, seq_length))
        return tf.tile(mask, tf.concat([
            tf.expand_dims(batch_size, -1), 
            tf.constant([1, 1], dtype=tf.int32)
        ], axis=0))


def TransformerEncoderBlock(
    receptive_size, # CNN output features size
    num_layers, # N encoder layers
    num_heads,
    embedding_dim, # d_model
    feed_forward_units, # dff
    dropout_rate,
    name = 'TransformerEncoderBlock'
):
    features_input = Input(shape=(receptive_size, embedding_dim), dtype='float32', name='cnn_features')
    embedding = TransformerEmbedding(embedding_dim, receptive_size, name='PositionalEmbedding')
    x = embedding(features_input)
    x = Dropout(dropout_rate, name='embedding_dropout')(x)

    for i in range(num_layers):
        x = TransformerEncoderLayer(
            num_heads = num_heads, 
            embedding_dim = embedding_dim, # d_model
            feed_forward_units = feed_forward_units, # dff
            dropout_rate = dropout_rate,
            name = f'Encoder{i + 1}'
        )(x)
    return tf.keras.Model(inputs=features_input, outputs=x, name=name)


def TransformerDecoderBlock(
    receptive_size, # CNN output features size
    seq_length, # = max_length - 1, cause the inputs is shifted by 1
    vocab_size,
    num_layers, # N decoder layers
    num_heads,
    embedding_dim, # d_model
    feed_forward_units, # dff
    dropout_rate,
    name = 'TransformerDecoderBlock'
):
    dec_seq_input = Input(shape=(seq_length,), dtype='int64', name='decoder_sequence')
    enc_output = Input(shape=(receptive_size, embedding_dim), dtype='float32', name='encoder_output')
    attention_weights = {}

    # Adding tokens and positional embedding
    embedding = TransformerEmbedding(embedding_dim, seq_length, vocab_size, name='TokenAndPosEmbedding')
    x = embedding(dec_seq_input)
    x = Dropout(dropout_rate, name='embedding_dropout')(x)

    for i in range(num_layers):
        layer_name = f'Decoder{i + 1}'
        x, block1, block2 = TransformerDecoderLayer(
            num_heads = num_heads, 
            embedding_dim = embedding_dim, # d_model
            feed_forward_units = feed_forward_units, # dff
            dropout_rate = dropout_rate,
            name = layer_name
        )(x, enc_output)
        attention_weights[f'{layer_name}_block1'] = block1
        attention_weights[f'{layer_name}_block2'] = block2

    y_pred = Dense(vocab_size, name='prediction')(x)
    return tf.keras.Model(inputs=[dec_seq_input, enc_output], outputs=[y_pred, attention_weights], name=name)


class TransformerOCR(EncoderDecoderModel):
    def __init__(self, cnn_model, encoder, decoder, data_handler, name='TransformerOCR', **kwargs):
        super(TransformerOCR, self).__init__(encoder, decoder, data_handler, name, **kwargs)
        self.cnn_model = cnn_model

    def get_config(self):
        config = super().get_config()
        config.update({'cnn_model': self.cnn_model})
        return config

    @tf.function
    def _compute_loss(self, batch):
        batch_images, batch_tokens = batch
        features = self.cnn_model(batch_images)
        enc_output = self.encoder(features)

        dec_seq_input = batch_tokens[:, :-1]
        dec_seq_real = batch_tokens[:, 1:]
        dec_seq_pred, _ = self.decoder([dec_seq_input, enc_output])
        loss = self.loss(dec_seq_real, dec_seq_pred)

        # Update training display result
        display_result = {'loss': loss}
        display_result = self.update_metrics(batch_images, batch_tokens, display_result)
        return loss, display_result

    @tf.function
    def predict(self, batch_images, return_attention=False):
        batch_size = batch_images.shape[0]
        seq_tokens = tf.fill([batch_size, 1], self.data_handler.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        result_tokens, attentions = [seq_tokens], []

        features = self.cnn_model(batch_images)
        enc_output = self.encoder(features)
        
        for _ in range(1, self.data_handler.max_length - 1):
            y_pred, attention_weights = self.decoder([seq_tokens, enc_output])
            attentions.append(attention_weights)

            # Select the last token from the seq_length (max_length - 1) dimension
            y_pred = y_pred[:, -1:, :] # (batch_size, 1, vocab_size)

            # Set the logits for all masked tokens to -inf, so they are never chosen
            y_pred = tf.where(self.data_handler.token_mask, float('-inf'), y_pred)
            new_tokens = tf.argmax(y_pred, axis=-1)

            # Once a sequence is done it only produces padding token
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)
            
            # Append new predicted tokens to the result sequences
            seq_tokens = tf.concat([seq_tokens, new_tokens], axis=-1)
            result_tokens.append(seq_tokens)

            # If a sequence produces an `END_TOKEN`, set it `done` after that
            done = done | (new_tokens == self.data_handler.end_token)
            if tf.executing_eagerly() and tf.reduce_all(done): break

        if return_attention: return result_tokens, tf.concat(attentions, axis=-1)
        return result_tokens
