============

- Dataset link:  https://www.openacademic.ai/oag/


Graph: Experiments on the Open Academic Graph (OAG) - CS dataset. We extract the citation graph from the origin heterogeneous graph. Each node is a paper and each directed edge indicates that one paper cites another one.  Each paper comes with a 768-dimensional feature vector obtained by the Bert Embedding of title. All papers are also associated with the year that the corresponding paper was published.

Prediction task: The task is to predict the subject areas of CS papers. The labels are extracted by the L1 field node in the origin OAG graph. Each paper may have multiple subject areas, which makes the task a multi-label classsification problem.

Dataset splitting: We train on papers published until 2017, validate on those published in 2018, and test on those published since 2019.

Dependencies
------------
- PyTorch 1.4.0+
- sklearn
- tqdm
- pandas
- numpy

How to run
----------

An experiment in default settings can be run with:

```bash
python train.py
```

Results
-------

Run with the default setting and we report the test-set performance of the model at the epoch when it has the best validation-set performance:

* Micro-F1: ~0.50, Exactly-Match-Ratio:~0.13 