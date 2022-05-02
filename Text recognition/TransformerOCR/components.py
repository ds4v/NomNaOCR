# https://www.tensorflow.org/text/tutorials/transformer
# https://keras.io/examples/nlp/neural_machine_translation_with_transformer
# https://keras.io/examples/vision/image_captioning
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Embedding, Dense, Dropout 


def point_wise_ffn(embedding_dim, feed_forward_units, dropout_rate=None):
    return tf.keras.Sequential([
        Dense(feed_forward_units, activation='relu'), # (batch_size, seq_len, feed_forward_units)
        Dropout(dropout_rate),
        Dense(embedding_dim) # (batch_size, seq_len, embedding_dim)
    ] if dropout_rate else [
        Dense(feed_forward_units, activation='relu'), # (batch_size, seq_len, feed_forward_units)
        Dense(embedding_dim) # (batch_size, seq_len, embedding_dim)
    ])


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_length, vocab_size, embedding_dim, name='PositionalEmbedding', **kwargs):
        ''' https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11/303
        We'll learn position-embedding vectors the same way we learn to embed word indices.
        We'll then proceed to add our position embeddings to the corresponding word embeddings, to
        obtain a position-aware word embedding. This technique is called "positional embedding".
        '''
        super(PositionalEmbedding, self).__init__(name=name, **kwargs)
        self.token_embeddings = Embedding(vocab_size, embedding_dim)
        self.position_embeddings = Embedding(max_length, embedding_dim)
        self.embed_scale = tf.sqrt(tf.cast(embedding_dim, tf.float32))


    def call(self, seq_input):
        positions = tf.range(start=0, limit=tf.shape(seq_input)[-1], delta=1)
        embedded_tokens = self.token_embeddings(seq_input) * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions


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
        super(EncoderLayer, self).__init__(name=name, **kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)
        self.ffn = point_wise_ffn(embedding_dim, feed_forward_units)

        # The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the
        # embedding_dim (last) axis. (epsilon: small float added to variance to avoid dividing by 0)
        self.layernorm1 = LayerNormalization(epsilon=1e-6) 
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)


    def call(self, features, training):
        attn_output, _ = self.mha(
            query=features, value=features, key=features,
            attention_mask=None, training=training
        ) # (batch_size, input_seq_len, embedding_dim)

        out1 = self.layernorm1(features + attn_output) # (batch_size, input_seq_len, embedding_dim)
        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, embedding_dim)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, embedding_dim)
        return out2


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
        super(DecoderLayer, self).__init__(name=name, **kwargs)
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)
        self.ffn = point_wise_ffn(embedding_dim, feed_forward_units)

        # The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the
        # embedding_dim (last) axis. (epsilon: small float added to variance to avoid dividing by 0)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(rate)


    def call(self, seq_tokens, enc_output, look_ahead_mask, padding_mask, training):
        # enc_output.shape == (batch_size, input_seq_len, embedding_dim)
        attn1, attn_weights_block1 = self.mha1(
            query=seq_tokens, value=seq_tokens, key=seq_tokens, 
            attention_mask=look_ahead_mask, training=training
        ) # (batch_size, target_seq_len, embedding_dim)
        out1 = self.layernorm1(attn1 + seq_tokens)

        attn2, attn_weights_block2 = self.mha2(
            query=out1, value=enc_output, key=enc_output, 
            attention_mask=padding_mask, training=training
        )  # (batch_size, target_seq_len, embedding_dim)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2) # (batch_size, target_seq_len, embedding_dim)
        ffn_output = self.dropout(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2) # (batch_size, target_seq_len, embedding_dim)
        return out3, attn_weights_block1, attn_weights_block2