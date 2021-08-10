# SASRecTF2

This contains implementation of SASRec using TF2 and python3.

The official code implemented in TF1: https://github.com/kang205/SASRec

### Dataset
The preprocessed Amazon Beauty dataset is provided in `./data`

For more robust comparison between methods, unlike previous code that randomly sample negative candidates for every epoch, we first sample 100 negative candidates and save it as `Beauty_test.json`.

We also provide negative candidates using popularity. i.e. sample top 100 popular items that are not in the user history. This is saved as `Beauty_test_pop.json`.

### Training
```
python train.py
```
See params in `train.py` for changing hyper-parameters.

### Make user & product model
To deploy your model with tf-serving, you need to separately save model into user and product model.
```
python make_serve_model.py
```

### Result
|             | HT@10  | NDCG@10 |
|-------------|--------|---------|
|TF1 test     | 0.5055 | 0.3333  |
|TF2 test     | 0.5080 | 0.3343  |
|TF2 test_pop | 0.1408 | 0.06242 |

Where, TF1 is result of official impli. with our test samples.

Result shows significant performance drop when using popularity based negative sampling.
