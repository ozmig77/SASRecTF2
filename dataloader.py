import tensorflow as tf
import json
import numpy as np

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def get_trainset(dataname, batch_size, max_len, buffer_size,
                 **kwargs):
    with open(f'./data/{dataname}_train.json', 'r') as f:
        anno = json.load(f)
    p_num = max([max(hist) for hist in anno]) + 1 # 0 ~ 12101, 12102

    def gen():
        for hist in anno:
            seq = np.zeros([max_len], dtype=np.int32)
            pos = np.zeros([max_len], dtype=np.int32)
            neg = np.zeros([max_len], dtype=np.int32)
            idx = max_len - 1
            nxt = hist[-1]

            ts = set(hist)
            for i in reversed(hist[:-1]):
                seq[idx] = i
                pos[idx] = nxt
                if nxt != 0: neg[idx] = random_neq(1, p_num-1, ts)
                nxt = i
                idx -= 1
                if idx == -1: break

            yield (seq, pos, neg)

    dataset = tf.data.Dataset.from_generator(
        lambda: gen(), (tf.int32, tf.int32, tf.int32))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=((None,),(None,),(None,)),
                                   padding_values=0)

    return dataset, p_num, (len(anno) // batch_size) + 1

def get_testset(dataname, max_len, batch_size, buffer_size,
                test_pop=False, **kwargs):

    if test_pop:
        print("** Load test based on popularity **")
        with open(f'./data/{dataname}_test_pop.json', 'r') as f:
            anno = json.load(f)
    else:
        with open(f'./data/{dataname}_test.json', 'r') as f:
            anno = json.load(f)

    def gen():
        for hist, tgt in anno:
            seq = np.zeros([max_len], dtype=np.int32)
            hist = hist[-max_len:] # Truncate to max_len
            seq[-len(hist):] = hist

            yield (seq, tgt)

    dataset = tf.data.Dataset.from_generator(
        lambda: gen(), (tf.int32, tf.int32))
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=((None,), (None,)),
                                   padding_values=0)

    return dataset, (len(anno) // batch_size) + 1

if __name__ == '__main__':
    # dataset, p_num = get_trainset(dataname='Beauty',
    #                               batch_size=2,
    #                               max_len=50,
    #                               buffer_size=10)
    dataset, a_len = get_testset(dataname='Beauty',
                                 max_len=50,
                                 buffer_size=10)
    print(next(dataset.as_numpy_iterator()))
