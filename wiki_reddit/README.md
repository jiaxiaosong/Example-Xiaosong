============

- Dataset link:  https://snap.stanford.edu/jodie/#code
- Dataset Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019.

We do node classification and link prediction task on the Wikipedia and Reddit dataset under the same setting with [1].

[1] Xu D, Ruan C, Korpeoglu E, et al. Inductive Representation Learning on Temporal Graphs. International Conference on Learning Representations (ICLR), 2020.


Dependencies
------------
- PyTorch 1.4.0+
- sklearn
- tqdm
- pandas
- numpy

How to run
----------

For the link prediction task, experiments in default settings can be run with:

```bash
python train_edge.py --data reddit
```

```bash
python train_edge.py --data wikipedia
```

For the node classification task, following [1], we pretrain the GNN model by the link prediction task and then only train a MLP for the feature extracted from the pretrained GNN. Experiments in default settings can be run with:

```bash
python train_node.py --data reddit --pretrained [snapshot_path]
```

```bash
python train_node.py --data wikipedia --pretrained [snapshot_path]
```

Note that in our code, during the training of link prediction task, the snapshot of GNN model has been saved. Please check the log folder.

Results
-------

Run with the default setting and we report the test-set performance of the model at the epoch when it has the best validation-set performance:


Link Prediction Task:

| Dataset | Transductive Accuracy | Transductive AP | Inductive Accuracy | Inductive AP  |
| ----- | ----------------- | ------------ | ------------------------------------ | ---------------------- |
| Reddit  | 0.9487 | 0.9903 | 0.9313 | 0.9832 | 
| Wikipedia  | 0.9221 | 0.9796 | 0.9191 | 0.9795 |


Node Classification Task:

| Dataset | AUC |
| ----- | ------|
| Reddit  | 0.7090 |
| Wikipedia  | 0.8011 | 