# https://www.tensorflow.org/text/tutorials/transformer
# https://keras.io/examples/nlp/neural_machine_translation_with_transformer
# https://keras.io/examples/vision/image_captioning
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
        seq_length, # For positional embedding
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
        self.positions_embedding = Embedding(seq_length, embedding_dim) if seq_length else None
        self.embed_scale = tf.sqrt(tf.cast(embedding_dim, tf.float32))

    def call(self, inputs, use_scale=True):
        seq_length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=seq_length, delta=1)
        embedded_positions = self.positions_embedding(positions)
        if self.tokens_embedding: inputs = self.tokens_embedding(inputs)
        if use_scale: inputs *= self.embed_scale
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        # This will be propagated to the mask param when call the TransformerDecoderLayer below
        return inputs != 0 # Not use padding token


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
        ffn_input = self.layernorm1(features + attn_output) 
        ffn_output = self.ffn(ffn_input)
        return self.layernorm2(out1 + ffn_output) # (batch_size, receptive_size, embedding_dim)


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
        look_ahead_mask = self.get_causal_attention_mask(inputs)
        padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
        padding_mask = tf.minimum(padding_mask, look_ahead_mask)

        attn1, attn_weights_block1 = self.mha1(
            query=inputs, value=inputs, key=inputs, 
            attention_mask=look_ahead_mask, return_attention_scores=True
        )  # (batch_size, max_length, embedding_dim)
        out1 = self.layernorm1(attn1 + inputs)

        attn2, attn_weights_block2 = self.mha2(
            query=out1, value=enc_output, key=enc_output, 
            attention_mask=padding_mask, return_attention_scores=True
        )  # (batch_size, max_length, embedding_dim)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2) # (batch_size, max_length, embedding_dim)
        out3 = self.layernorm3(ffn_output + out2) # (batch_size, max_length, embedding_dim)
        return out3, attn_weights_block1, attn_weights_block2

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, max_length = input_shape[0], input_shape[1]
        seq_range = tf.range(max_length)
        
        mask = tf.cast(seq_range[:, tf.newaxis] >= seq_range, dtype=tf.int32)
        mask = tf.reshape(mask, (1, max_length, max_length))
        return tf.tile(mask, tf.concat([
            tf.expand_dims(batch_size, -1), 
            tf.constant([1, 1], dtype=tf.int32)
        ], axis=0))


def TransformerEncoderBlock(
    cnn_features_shape,
    num_layers, # N encoder layers
    num_heads,
    embedding_dim, # d_model
    feed_forward_units, # dff
    dropout_rate,
    name = 'TransformerEncoderBlock'
):
    features_input = Input(shape=cnn_features_shape, dtype='float32', name='cnn_features')
    receptive_size = tf.shape(features_input)[1]
    x = TransformerEmbedding(embedding_dim, receptive_size)(features_input)
    # x = LayerNormalization(epsilon=1e-6)(features_input)
    # x = Dense(embedding_dim, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    for _ in range(num_layers):
        x = TransformerEncoderLayer(num_heads, embedding_dim, feed_forward_units, dropout_rate)(x)
    return tf.keras.Model(inputs=features_input, outputs=x, name=name)


def TransformerDecoderBlock(
    vocab_size,
    num_layers, # N decoder layers
    num_heads,
    embedding_dim, # d_model
    feed_forward_units, # dff
    dropout_rate,
    name = 'TransformerDecoderBlock'
):
    dec_seq_input = Input(shape=(None,), dtype='int64', name='decoder_sequence')
    max_length = tf.shape(dec_seq_input)[1]

    enc_output = Input(shape=(max_length, embedding_dim), dtype='float32', name='encoder_output')
    attention_weights = {}

    # Adding embedding and position encoding
    x = TransformerEmbedding(embedding_dim, max_length, vocab_size)(dec_seq_input)
    x = Dropout(dropout_rate)(x)

    for i in range(num_layers):
        decoder = TransformerDecoderLayer(num_heads, embedding_dim, feed_forward_units, dropout_rate)
        x, attn_weights_block1, attn_weights_block2 = decoder(x, enc_output)
        attention_weights[f'decoder_layer{i + 1}_block1'] = attn_weights_block1
        attention_weights[f'decoder_layer{i + 1}_block2'] = attn_weights_block2

    y_pred = Dense(vocab_size, name='prediction')(x)
    return tf.keras.Model(inputs=[dec_seq_input, enc_output], outputs=[y_pred, attention_weights], name=name)


class TransformerOCR(EncoderDecoderModel):
    def __init__(self, encoder, decoder, data_handler, name='TransformerOCR'):
        super(TransformerOCR, self).__init__(encoder, decoder, data_handler, name)

    def call(self, inputs, training=None):
        pass