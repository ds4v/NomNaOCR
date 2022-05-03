# https://www.tensorflow.org/text/tutorials/transformer
# https://keras.io/examples/nlp/neural_machine_translation_with_transformer
# https://keras.io/examples/vision/image_captioning
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout,
    MultiHeadAttention, LayerNormalization
)


def point_wise_ffn(embedding_dim, feed_forward_units, dropout_rate=None):
    return tf.keras.Sequential([
        Dense(feed_forward_units, activation='relu'), # (batch_size, seq_len, feed_forward_units)
        Dropout(dropout_rate),
        Dense(embedding_dim) # (batch_size, seq_len, embedding_dim)
    ] if dropout_rate else [
        Dense(feed_forward_units, activation='relu'), # (batch_size, seq_len, feed_forward_units)
        Dense(embedding_dim) # (batch_size, seq_len, embedding_dim)
    ])


class TransformerEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim, # d_model
        max_length, # For positional embedding
        vocab_size = None, # If not None, tokens embedding will be used
        name = 'TransformerEmbedding',
        **kwargs
    ):
        ''' https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11/303
        We'll learn position-embedding vectors the same way we learn to embed word indices.
        We'll then proceed to add our position embeddings to the corresponding word embeddings, to
        obtain a position-aware word embedding. This technique is called "positional embedding".
        '''
        super(TransformerEmbedding, self).__init__(name=name, **kwargs)
        self.tokens_embedding = Embedding(vocab_size, embedding_dim) if vocab_size else None
        self.positions_embedding = Embedding(max_length, embedding_dim) if max_length else None
        self.embed_scale = tf.sqrt(tf.cast(embedding_dim, tf.float32))

    def call(self, inputs, use_scale=True):
        max_length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=max_length, delta=1)
        embedded_positions = self.positions_embedding(positions)
        if self.tokens_embedding: inputs = self.tokens_embedding(inputs)
        if use_scale: inputs *= self.embed_scale
        return inputs + embedded_positions # (batch_size, max_length, embedding_dim)


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
        self.ffn = point_wise_ffn(embedding_dim, feed_forward_units)

        # The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the
        # embedding_dim (last) axis. (epsilon: small float added to variance to avoid dividing by 0)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)

    def call(self, features):
        attn_output, _ = self.mha(query=features, value=features, key=features, attention_mask=None)
        out1 = self.layernorm1(features + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2  # (batch_size, max_length, embedding_dim)


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
    # max_length = tf.shape(features_input)[1]
    # x = TransformerEmbedding(embedding_dim, max_length)(features_input)
    x = LayerNormalization(epsilon=1e-6)(features_input)
    x = Dense(embedding_dim, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    for _ in range(num_layers):
        x = TransformerEncoderLayer(num_heads, embedding_dim, feed_forward_units, dropout_rate)(x)
    return tf.keras.Model(inputs=features_input, outputs=x, name=name)


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
        self.ffn = point_wise_ffn(embedding_dim, feed_forward_units)

        # The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the
        # embedding_dim (last) axis. (epsilon: small float added to variance to avoid dividing by 0)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)

    def call(self, seq_tokens, enc_output, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, max_length, embedding_dim)
        attn1, attn_weights_block1 = self.mha1(
            query=seq_tokens, value=seq_tokens, key=seq_tokens, attention_mask=look_ahead_mask
        )  # (batch_size, max_length, embedding_dim)
        out1 = self.layernorm1(attn1 + seq_tokens)

        attn2, attn_weights_block2 = self.mha2(
            query=out1, value=enc_output, key=enc_output, attention_mask=padding_mask
        )  # (batch_size, max_length, embedding_dim)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2) # (batch_size, max_length, embedding_dim)
        ffn_output = self.dropout(ffn_output)
        out3 = self.layernorm3(ffn_output + out2) # (batch_size, max_length, embedding_dim)
        return out3, attn_weights_block1, attn_weights_block2


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

    # Adding embedding and position encoding.
    x = TransformerEmbedding(embedding_dim, max_length, vocab_size)(dec_seq_input)
    x = Dropout(dropout_rate)(x)

    for i in range(num_layers):
        x, block1, block2 = TransformerDecoderLayer(
            num_heads, embedding_dim, feed_forward_units, dropout_rate
        )(x, enc_output, look_ahead_mask, padding_mask)
        attention_weights[f'decoder_layer{i + 1}_block1'] = block1
        attention_weights[f'decoder_layer{i + 1}_block2'] = block2
    return tf.keras.Model(inputs=[dec_seq_input, enc_output], outputs=[x, attention_weights], name=name)


def create_padding_mask(seq):
    seq = tf.cast(seq == 0, tf.float32)
    # Add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, max_length)

def create_look_ahead_mask(max_length):
    mask = 1 - tf.linalg.band_part(tf.ones((max_length, max_length)), -1, 0)
    return mask # (max_length, max_length)
