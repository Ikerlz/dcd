# `dcd`: distributed spectral clustering

**implemented with Apache Spark**

## Introduction

In this work we develop a distributed spectral clustering algorithm for community detection in large scale networks. To handle the problem, we distribute l pilot network nodes on the master server and the others on worker servers. A spectral clustering algorithm is first conducted on the master to select pseudo centers. The indexes of the pseudo centers are then broadcasted to workers to complete distributed community detection task using a SVD type algorithm. The proposed distributed algorithm has three merits. First, the communication cost is low since only the indexes of pseudo centers are communicated. Second, no further iteration algorithm is needed on workers and hence it does not suffer from problems as initialization and non-robustness. Third, both the computational complexity and the storage requirements are much lower compared to using the whole adjacency matrix. A Python package [DCD](www.github.com/Ikerlz/dcd) is developed to implement the distributed algorithm for a Spark system. Theoretical properties are provided with respect to the estimation accuracy and mis-clustering rates. Lastly, the advantages of the proposed methodology are illustrated by experiments on a variety of synthetic and empirical datasets.

-----

## Dependencies

- `Spark >= 2.3.1`
- `Python >= 3.7.0`
  - `pyarrow >= 0.15.0` Please read this [compatible issue with Spark 2.3.x or 2.4.x](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#compatibility-setting-for-pyarrow--0150-and-spark-23x-24x)
  - `numpy >= 1.16.3`
  - `findspark >= 1.3.0`

-----

## Run the PySpark code on the Spark platform

### Simulation

```
python simulation.py
```

### Real Data

- AgBlog

```
python AGBlog/main.py
```

- Pumbed

```
python Pumbed/main.py
```

- Pokec

```
python Pokec/main.py
```

- Cora

```
python Cora/main.py
```

### Norm based method

This is an implement of [A Divide and Conquer Framework for Distributed Graph Clustering](http://proceedings.mlr.press/v37/yange15.pdf)

```
python DC_method/main.py
```

## Comparison

Compare **dcd** with the **norm based method**

```
python Comparison/main.py
```


-----

## License
[MIT License](https://github.com/Ikerlz/dcd/blob/master/LICENSE)


-----

## References

[Li, Z.](http://lizhe.fun/), & [Wu, S.]() [Zhu, X.](https://xueningzhu.github.io/), (2020) Distributed Community Detection for Large Scale Networks Using Stochastic Block Model. [_Working Paper_](https://arxiv.org/pdf/2009.11747.pdf).
