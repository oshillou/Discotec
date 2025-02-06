# Discriminative ordering through ensemble consensus

This repository contains all code to reproduce the experiments for the "Discriminative ordering through ensemble consensus".

Most of the code for the experiments is contained in jupyter notebooks, with one dedicated folder per experiment. The figures and tables and baselines are produced by higher level notebooks at the root folder.

For the core code of the method, you may find it in the `discotec.py` file.

## Synthetic experiments

These experiments correspond to section 4.2 of the paper. They are in the `Synthetic partitions` with one sub-folder per scenario. Each contains a jupter notebook that produce their respective results and figures.

## Benchmark with UCI and FCPS dataset

This corresponds to section 4.3 of the paper.

### Benchmark with as many clusters as possible

The benchmark with FCPS dataset can be found in the `Synthetic folder`, where all data is available. Running the notebook `Synthetic datasets clustering` will generate all necessary clusterings for that experiment. The baselines can then be recovered using the root-level notebook `Baselines` (*Do not forget to adapt the path to the correct folder*). Finally, the tables of the experiment can be reconstructed using the `Table constructor` notebook.

The same thing can be done by using the `UCI datasets` folder for UCI-related experiment,

### Benchmark with a fixed number of cluster

It is the same protocol as the previous experiment, except folders are suffixed with *(same number of clusters)*.

## Constraint additions

This experiment from section 4.4 can be reproduced using the root-level notebook `Integrating constraints`.
