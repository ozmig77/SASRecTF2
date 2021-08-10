import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np

class PositionalEncoding(layers.Layer):
    def __init__(self, position, dims):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, dims)

    def get_angles(self, position, i, dims):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(dims, tf.float32))
        return position * angles

    def positional_encoding(self, position, dims):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis], # P x 1
            i=tf.range(dims, dtype=tf.float32)[tf.newaxis, :], # 1 x D
            dims=dims
        )

        # 짝수는 sin, 홀수는 cos 적용
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads, dtype=tf.float32)
        pos_encoding = pos_encoding[tf.newaxis, ...] # 1 x P x D

        return pos_encoding

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def create_look_ahead_mask(size):
    '''
    when size = 3, return
    0 1 1
    0 0 1
    0 0 0
    '''
    ones = tf.ones((size, size))
    mask = 1 - tf.linalg.band_part(ones, -1, 0)
    return mask # (size, size)

def create_mask(mask):
    '''
    Args
      mask: (batch_size, key_len), if 1 then mask
    '''
    look_ahead_mask = create_look_ahead_mask(tf.shape(mask)[1]) # K x K
    look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, ...] # 1 x 1 x K x K

    mask = mask[:, tf.newaxis, tf.newaxis, :] # B x 1 x 1 x K

    mask = tf.maximum(mask, look_ahead_mask) # B x 1 x K x K
    return mask


def scaled_dot_product_attention(query, key, value, mask):
    '''
    Args
      query: (batch, num_heads, query_len, dims/num_heads)
      key: (batch, num_heads, key_len, dims/num_heads)
      value: (batch, num_heads, value_len, dims/num_heads)
      mask: (batch_size, 1, key_len, key_len), if 1 then mask
    '''
    depth = tf.cast(tf.shape(key)[-1], tf.float32)

    # Sacled dot product
    matmul_qk = tf.matmul(query, key, transpose_b=True) # B x H x Q x K
    logits = matmul_qk / tf.math.sqrt(depth)

    # Masking
    if mask is not None:
        logits += (mask * -1e9)

    # Softmax
    attention_weights = tf.nn.softmax(logits, axis=-1) # B x H x Q x K

    # Output
    output = tf.matmul(attention_weights, value) # B x H x Q x D//H


    return output, attention_weights

class MultiHeadAttention(layers.Layer):
    def __init__(self, dims, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dims = dims
        assert dims % num_heads == 0

        self.depth = dims // num_heads

        self.wq = layers.Dense(dims)
        self.wk = layers.Dense(dims)
        self.wv = layers.Dense(dims)

        #self.dense = layers.Dense(dims)

    def split_heads(self, x, batch_size):
        '''
        Args
          x: (batch, len, dims)
        '''
        x = tf.reshape(x,
            (batch_size, -1, self.num_heads, self.depth)
        ) # B x L x H x D//H
        return tf.transpose(x, perm=[0,2,1,3]) # B x H x L x D//H

    def call(self, v, k, q, mask):
        '''
        Args
          v: (batch, value_len, dims)
          k: (batch, key_len, dims)
          q: (batch, query_len, dims)
          mask: (batch, 1, key_len, key_len)
        '''
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size) # B x H x Q x D//H
        k = self.split_heads(k, batch_size) # B x H x K x D//H
        v = self.split_heads(v, batch_size) # B x H x V x D//H

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        ) # B x H x Q x D//H, B x H x Q x K
        scaled_attention = tf.transpose(scaled_attention,
                                perm=[0, 2, 1, 3])  # B x Q x H x D//H
        concat_attention = tf.reshape(scaled_attention,
                                (batch_size, -1, self.dims))  # B x Q x D
        output = concat_attention
        #output = self.dense(concat_attention)  # B x Q x D

        return output, attention_weights

def point_wise_feed_forward_network(dims, dff, rate):
  return tf.keras.Sequential([
      layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      layers.Dropout(rate),
      layers.Dense(dims),  # (batch_size, seq_len, dims)
      layers.Dropout(rate)
  ])

class MaskedEncoderLayer(layers.Layer):
    def __init__(self, dims, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(dims, num_heads)
        self.ffn = point_wise_feed_forward_network(dims, dff, rate)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-8)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-8)

        self.dropout1 = layers.Dropout(rate)

    def call(self, x, mask, training):
        '''
        Args
          x: (batch_size, seq_len, dims)
          training: boolean
          mask: (batch_size, seq_len)
        '''
        mask = create_mask(mask) # B x 1 x L x L
        attn_output, _ = self.mha(x, x, x, mask)  # B x L x D
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # B x L x D

        ffn_output = self.ffn(out1, training=training)  # B x L x D
        out2 = self.layernorm2(out1 + ffn_output)  # B x L x D

        return out2


if __name__ == '__main__':
    #mask = tf.constant([[0,0,0,1],
    #                 [0,0,1,1.]])
    #print(create_mask(mask))

    seq = layers.Input(shape=(None, 16))
    masks = layers.Input(shape=(None,)) # L
    encoder = MaskedEncoderLayer(dims=16, num_heads=2, dff=32)
    outputs = encoder(seq, masks)
    model = Model(inputs=(seq,masks),
                  outputs = outputs)
    print(model.layers[-1].weights)
    #print(model.layers[-1].get_weights())








#
