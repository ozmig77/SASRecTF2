import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import L2
from transformer import MaskedEncoderLayer, PositionalEncoding
import numpy as np

class SASRec(Model):
    def __init__(self, p_num, hidden_dim,
                 max_len=50, rate=0.2, l2_reg=0.0,
                 sas_poe='dynamic', sas_agg='last',
                 num_blocks=2, num_heads=1, **kwargs):
        super().__init__()
        self.poe = sas_poe # none, static, dynamic
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks


        self.product_embedding = layers.Embedding(p_num,
                                        hidden_dim,
                                        input_length=None,
                                        embeddings_initializer=GlorotUniform(),
                                        embeddings_regularizer=L2(l2_reg),
                                        mask_zero=True,
                                        name="product_embedding")
        if self.poe == 'static':
            self.poe_layer = PositionalEncoding(max_len, hidden_dim)
        elif self.poe == 'dynamic':
            self.poe_layer = layers.Embedding(max_len,
                                    hidden_dim,
                                    input_length=None,
                                    embeddings_initializer=GlorotUniform(),
                                    embeddings_regularizer=L2(l2_reg),
                                    name="position_embedding")
        self.user_sas = [MaskedEncoderLayer(
            dims=hidden_dim,
            num_heads=num_heads,
            dff=hidden_dim,
            rate=rate) for _ in range(self.num_blocks)]

        #self.dropout_p = layers.Dropout(rate)
        self.dropout_u = layers.Dropout(rate)
        #self.layernorm_p = layers.LayerNormalization(axis=-1)
        #self.layernorm_u = layers.LayerNormalization(axis=-1)
        self.dots = layers.Dot(axes=(1,1), normalize=False)

    def user_model(self, inputs, training=None):
        '''
        inputs
            seq: Int, B x L
        '''
        seq = inputs
        mask = tf.expand_dims(tf.cast(tf.not_equal(seq, 0), tf.float32), -1) # B x L x 1

        hist_emb = self.product_embedding(seq) # B x L x E
        hist_emb = hist_emb * (self.hidden_dim ** 0.5)

        if self.poe == 'static':
            hist_emb = self.poe_layer(hist_emb)
        elif self.poe == 'dynamic':
            poe = tf.expand_dims(tf.range(tf.shape(hist_emb)[1]), 0) # 1 x L
            poe = self.poe_layer(poe) # 1 x L x E
            hist_emb += poe

        hist_emb = self.dropout_u(hist_emb, training=training) # B x L x E
        hist_emb = hist_emb * mask

        sas_mask = tf.squeeze(mask, axis=-1) # B x L
        sas_mask = 1 - sas_mask # if 1 then mask

        for user_sas in self.user_sas:
            hist_emb = user_sas(hist_emb, mask=sas_mask, training=training) # B x L x E
            hist_emb *= mask
#        whole_output = self.user_sas2(whole_output, mask=sas_mask, training=training) # B x L x E
#        whole_output *= mask

        return hist_emb

    def product_model(self, inputs, training=None):
        '''
        inputs
            product: Int, B x L
        '''
        product = inputs

        product_emb = self.product_embedding(product) # B x L x E
        #product_emb = self.layernorm_p(product_emb)
        #product_emb = self.dropout_p(product_emb, training=training) # B x E

        return product_emb

    def call(self, inputs, training):
        '''
        inputs
            seq: Int, B x L
            pos: Int, B x L
            neg: Int, B x L
        '''

        seq, pos, neg = inputs
        B = tf.shape(seq)[0]
        L = tf.shape(seq)[1]

        # --- User embedding ------
        seq_emb = self.user_model(seq, training=training) # B x L x E

        # --- Product embedding ------
        pos_emb = self.product_model(pos, training=training) # B x L x E
        neg_emb = self.product_model(neg, training=training) # B x L x E

        # --- Calculate score ------
        seq_emb = tf.reshape(seq_emb, (B*L, self.hidden_dim)) # B*L x E
        pos_emb = tf.reshape(pos_emb, (B*L, self.hidden_dim)) # B*L x E
        neg_emb = tf.reshape(neg_emb, (B*L, self.hidden_dim)) # B*L x E
        istarget = tf.reshape(tf.cast(tf.not_equal(pos, 0), tf.float32),
                                          [B * L]) # B*L

        logits2 = tf.reduce_sum(seq_emb * pos_emb, -1)

        pos_logits = tf.squeeze(self.dots([seq_emb, pos_emb]), -1) # B*L
        neg_logits = tf.squeeze(self.dots([seq_emb, neg_emb]), -1) # B*L

        # --- Calculate loss ------
        loss = tf.reduce_sum(
            - tf.math.log(tf.sigmoid(pos_logits) + 1e-24) * istarget -
            tf.math.log(1 - tf.sigmoid(neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)

        reg_losses=tf.add_n(self.losses)
        loss += reg_losses

        return loss # 1

    def test(self, inputs, training=False):
        '''
        inputs
            seq: Int, B x L
            tgt: Int, B x 101
        '''
        seq, tgt = inputs
        B = tf.shape(seq)[0]
        L = tf.shape(seq)[1]

        # --- User embedding ------
        seq_emb = self.user_model(seq, training=training) # B x L x E
        seq_emb = seq_emb[:,-1:] # B x 1 x E

        # --- Product embedding ------
        tgt_emb = self.product_model(tgt, training=training) # B x 101 x E

        tgt_logits = tf.reduce_sum(seq_emb * tgt_emb, -1) # B x 101

        return tgt_logits

def build_rm_model(config):
    seq = layers.Input(shape=(None,), name='seq')
    pos = layers.Input(shape=(None,), name='pos')
    neg = layers.Input(shape=(None,), name='neg')
    tgt = layers.Input(shape=(None,), name='tgt')
    if config['model'] == 'SAS':
        rm = SASRec(**config)

    inputs = (seq, pos, neg)
    model = Model(inputs=inputs,
                  outputs = rm(inputs)) # 1

    inputs = (seq, tgt)
    test_model = Model(inputs = inputs,
                  outputs = rm.test(inputs)) # B x 101

    inputs = seq
    user_model = Model(inputs=inputs,
                  outputs = rm.user_model(inputs)[:,-1]) # B x E
    inputs = tgt
    product_model = Model(inputs=inputs,
                  outputs = rm.product_model(inputs)) # B x L x E

    return (model, test_model, user_model, product_model)









#
