# https://www.tensorflow.org/text/tutorials/transformer
# https://keras.io/examples/nlp/neural_machine_translation_with_transformer
# https://keras.io/examples/vision/image_captioning
# https://medium.com/geekculture/scene-text-recognition-using-resnet-and-transformer-c1f2dd0e69ae
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Add, MultiHeadAttention, LayerNormalization
from models import CustomTrainingModel


def point_wise_ffn(embedding_dim, feed_forward_units):
    return tf.keras.Sequential([
        Dense(feed_forward_units, activation='relu'), # (batch_size, seq_length, feed_forward_units)
        Dense(embedding_dim) # (batch_size, seq_length, embedding_dim)
    ]) 


class TransformerEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim, # d_model
        seq_length, # For positional embedding => receptive_size if features or max_length - 1 if texts
        vocab_size = None, # If None => Disable tokens embedding, just use positional embedding
        use_scale = True,
        use_pos_embed = False,
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
        self.positions_embedding = Embedding(seq_length, embedding_dim) if use_pos_embed else None
        self.embed_scale = tf.sqrt(tf.cast(embedding_dim, tf.float32)) if use_scale else None
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length


    def call(self, inputs):
        if self.positions_embedding: 
            positions = tf.range(start=0, limit=self.seq_length, delta=1)
            positions_info = self.positions_embedding(positions)
        else: positions_info = self.positional_encoding()

        if self.tokens_embedding: inputs = self.tokens_embedding(inputs)
        if self.embed_scale is not None: inputs *= self.embed_scale
        return inputs + positions_info


    def positional_encoding(self):
        pos = np.arange(self.seq_length)[:, np.newaxis]
        i = np.arange(self.embedding_dim)[np.newaxis, :]
        angle_rads = pos / np.power(10000, (2 * (i//2)) / np.float32(self.embedding_dim))
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) # Apply sin to even indices in the array; 2i
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) # Apply cos to odd indices in the array; 2i+1
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)


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
        self.ffn = point_wise_ffn(embedding_dim, feed_forward_units)

        # The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the
        # embedding_dim (last) axis. (epsilon: small float added to variance to avoid dividing by 0)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.supports_masking = True


    def call(self, features, training, mask=None):
        attn_output = self.mha(query=features, value=features, key=features, attention_mask=None, training=training)
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
        self.ffn = point_wise_ffn(embedding_dim, feed_forward_units)

        # The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the
        # embedding_dim (last) axis. (epsilon: small float added to variance to avoid dividing by 0)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.supports_masking = True


    def call(self, inputs, enc_output, training, mask=None):
        # enc_output.shape == (batch_size, receptive_size, embedding_dim)
        padding_mask = None
        if mask is not None: # Check for the attention mask when use encoder output in MultiHeadAttention
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)

        causal_mask = self.get_causal_attention_mask(inputs)
        look_ahead_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
        look_ahead_mask = tf.minimum(look_ahead_mask, causal_mask)

        attn1, attn_weights_block1 = self.mha1(
            query=inputs, value=inputs, key=inputs, training=training,
            attention_mask=look_ahead_mask, return_attention_scores=True
        ) # (batch_size, max_length - 1, embedding_dim)
        out1 = self.layernorm1(attn1 + inputs)

        attn2, attn_weights_block2 = self.mha2(
            query=out1, value=enc_output, key=enc_output, training=training, 
            attention_mask=padding_mask, return_attention_scores=True
        ) # (batch_size, max_length - 1, embedding_dim)
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
    use_skip_connection = False,
    name = 'TransformerEncoderBlock'
):
    features_maps = Input(shape=(receptive_size, embedding_dim), dtype='float32', name='cnn_features')
    x = TransformerEmbedding(embedding_dim, receptive_size, name='PositionalEncoding')(features_maps)
    if dropout_rate > 0: x = Dropout(dropout_rate, name='embedding_dropout')(x)
    
    for i in range(num_layers):
        x = TransformerEncoderLayer(
            num_heads = num_heads, 
            embedding_dim = embedding_dim, # d_model
            feed_forward_units = feed_forward_units, # dff
            dropout_rate = dropout_rate,
            name = f'Encoder{i + 1}'
        )(x)

    if use_skip_connection: x = Add(name='skip_connection')([features_maps, x])
    return tf.keras.Model(inputs=features_maps, outputs=x, name=name)


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

    # Adding tokens embedding and positional encoding
    x = TransformerEmbedding(embedding_dim, seq_length, vocab_size, name='TokEmbAndPosEncode')(dec_seq_input)
    if dropout_rate > 0: x = Dropout(dropout_rate, name='embedding_dropout')(x)

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


class TransformerOCR(CustomTrainingModel):
    def __init__(self, cnn_model, encoder, decoder, data_handler, name='TransformerOCR', **kwargs):
        super(TransformerOCR, self).__init__(data_handler, name, **kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder


    def get_config(self):
        return {
            'cnn_model': clone_model(self.cnn_model), 
            'encoder': clone_model(self.encoder), 
            'decoder': clone_model(self.decoder),
            'data_handler': self.data_handler
        }


    @tf.function
    def _compute_loss_and_metrics(self, batch, is_training=False):
        batch_images, batch_tokens = batch
        features = self.cnn_model(batch_images, training=is_training) # (batch_size, receptive_size, embedding_dim)
        enc_output = self.encoder(features, training=is_training) if self.encoder else features 

        dec_seq_input = batch_tokens[:, :-1]
        dec_seq_real = batch_tokens[:, 1:]
        dec_seq_pred, _ = self.decoder([dec_seq_input, enc_output], training=is_training) # (batch_size, seq_length, embedding_dim)
        
        loss = self.loss(dec_seq_real, dec_seq_pred)
        metrics = self._update_metrics(batch)
        return loss, {'loss': loss, **metrics}


    @tf.function
    def predict(self, batch_images, return_attention=False):
        batch_size = batch_images.shape[0]
        seq_tokens, done = self._init_seq_tokens(batch_size, return_new_tokens=False)
        features = self.cnn_model(batch_images, training=False) # (batch_size, receptive_size, embedding_dim)
        enc_output = self.encoder(features, training=False) if self.encoder else features
        attentions = []
        
        for i in range(1, self.data_handler.max_length):
            y_pred, attention_weights = self.decoder([seq_tokens[:, :-1], enc_output], training=False)
            attentions.append(attention_weights)
            y_pred = y_pred[:, i - 1, :] # Select last token from seq_length (max_length - 1) dimension (batch_size, 1, vocab_size)
            seq_tokens, done = self._update_seq_tokens(y_pred, seq_tokens, done, i, return_new_tokens=False)
            if tf.executing_eagerly() and tf.reduce_all(done): break

        if not return_attention: return seq_tokens
        return seq_tokens, attentions