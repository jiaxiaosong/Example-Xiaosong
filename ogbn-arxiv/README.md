============

- Dataset link:  https://ogb.stanford.edu/docs/home/
- Dataset Paper: Hu W, Fey M, Zitnik M, et al. Open graph benchmark: Datasets for machine learning on graphs[J]. arXiv preprint arXiv:2005.00687, 2020.

We do experiments on the ogbn-arxiv of Open Graph Benchmark (OGB). Please see the its official page and paper for the details of dataset.

Dependencies
------------
- PyTorch 1.4.0+
- sklearn
- tqdm
- pandas
- numpy
- ogb

How to run
----------

An experiment in default settings can be run with:

```bash
python train.py
```

Results
-------

Run with the default setting and we report the test-set performance of the model at the epoch when it has the best validation-set performance:

* Accuracy: ~0.71