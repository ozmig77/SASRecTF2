import tensorflow as tf
from tensorflow import keras
from model import build_rm_model

import argparse
from tqdm import tqdm
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default="beauty_sas_d16",
                    help='tag name to make user and product models')
parser.add_argument('--test', action="store_true",
                    help='if provided, perform testing')
args = parser.parse_args()


config_path = f"ckpts/{args.tag}/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Build model
model, _, user_model, product_model = build_rm_model(config)

# Load trained ckpts
loaded_model = keras.models.load_model(f'ckpts/{args.tag}/model')
model.set_weights(loaded_model.get_weights())

# Save user & product model
user_model.save(f'ckpts/{args.tag}/user_model/1')
product_model.save(f'ckpts/{args.tag}/product_model/1')

if args.test:
    from dataloader import get_testset
    test_dataset, test_len = get_testset(**config)
    # --- Evaluate -----
    NDCG, HT, valid_user = 0., 0., 0.
    for seq, tgt in tqdm(test_dataset, ncols=80, total=test_len):
        # --- User embedding ------
        seq_emb = user_model(seq, training=False) # B x E
        seq_emb = tf.expand_dims(seq_emb, 1)

        # --- Product embedding ------
        tgt_emb = product_model(tgt, training=False) # B x 101 x E
        preds = tf.reduce_sum(seq_emb * tgt_emb, -1) # B x 101


        for pred in preds.numpy():
            pred = -1 * pred
            rank = pred.argsort().argsort()[0]
            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
    NDCG, HT = NDCG/valid_user, HT/valid_user
    print(f"NDCG@10: {NDCG}, HT@10: {HT}")
