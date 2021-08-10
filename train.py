import os
import argparse
import numpy as np
import random
import datetime
import json
import tensorflow as tf

from dataloader import get_trainset, get_testset
from model import build_rm_model
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', default="Beauty",
                    help='data name')
parser.add_argument('--test_pop', action="store_true",
                    help='if proviede, use popularity based test samples')
parser.add_argument('--model', default="SAS",
                    help='model name')
parser.add_argument('--hidden_dim', default=16,
                    help='hidden dimension')
parser.add_argument('--rate', default=0.5,
                    help='dropout rate')
parser.add_argument('--lr', default=1e-3,
                    help='learning rate')
parser.add_argument('--l2_reg', default=1e-4,
                    help='weight decay')
parser.add_argument('--sas_poe', default="dynamic",
                    help='self attention position embedding, dynamic or static')
parser.add_argument('--num_block', default=2,
                    help='self attention number of layers')
parser.add_argument('--num_heads', default=1,
                    help='self attention number of heads')
parser.add_argument('--max_len', default=50,
                    help='max number of history sequences')
parser.add_argument('--batch_size', default=128,
                    help='batch size')
parser.add_argument('--buffer_size', default=10000,
                    help='buffer size for dataloader')
parser.add_argument('--epochs', default=200,
                    help='epochs to learn')
parser.add_argument('--seed', default=2021,
                    help='seed')
parser.add_argument('--tag', default="beauty_sas_d16",
                    help='tag name to make user and product models')
args = parser.parse_args()

# Set seed
tf.random.set_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# config
config = args.__dict__
tag = args.tag

print(f"***** Training {tag} *****")

# Load dataset
train_dataset, p_num, train_len = get_trainset(**config)
test_dataset, test_len = get_testset(**config)
config['p_num'] = p_num # product_num

# Save config
if not os.path.exists(f"ckpts/{tag}"):
    os.makedirs(f"ckpts/{tag}")
config_path = f"ckpts/{tag}/config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# Set up log
log_path = f"ckpts/{tag}/log.txt"
with open(log_path, 'w') as f:
    pass

# Build model
model, test_model, _, _ = build_rm_model(
    config)

print(model.layers[-1].summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'],
                                     beta_2=0.98)

@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        loss = model(inputs, training=True)

    # Compute gradients
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # Update weights
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss

best_res = 0
for epoch in range(config['epochs']):
    t = tqdm(train_dataset, ncols=80, total=train_len, leave=False)
    for i, inputs in enumerate(t): # seq, pos, neg
        loss = train_step(inputs)

        if i % 50 == 0:
            t.set_description(f"{epoch}: {loss}")

    if epoch % 20 != 19:
        continue
    # --- Evaluate -----
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    for seq, tgt in tqdm(test_dataset, ncols=80, total=test_len):
        preds = test_model((seq, tgt), training=False) # B x 101

        for pred in preds.numpy():
            pred = -1 * pred
            rank = pred.argsort().argsort()[0]
            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
    NDCG, HT = NDCG/valid_user, HT/valid_user
    res_str = f"Epoch: {epoch}, NDCG@10: {NDCG}, HT@10: {HT}"
    print(res_str)
    with open(log_path, 'a') as f:
        f.write(res_str + '\n')

    if HT > best_res:
        best_res = HT
        # Save ckpts
        model.save(f'ckpts/{tag}/model')
