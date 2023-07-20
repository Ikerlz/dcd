# `dcd`: distributed spectral clustering

**implemented with Apache Spark**

## Introduction

Community detection for large scale networks is of great importance in modern data analysis. In this work, we develop a distributed spectral clustering algorithm to handle this task. Specifically, we distribute a certain number of pilot network nodes on the master server and the others on worker servers. A spectral clustering algorithm is first conducted on the master to select pseudo centers. Next, the indexes of the pseudo centers are broadcasted to workers to complete the distributed community detection task using an SVD (singular value decomposition) type algorithm. The proposed distributed algorithm has three advantages. First, the communication cost is low, since only the indexes of pseudo centers are communicated. Second, no further iterative algorithm is needed on workers while a “one-shot” computation suffices. Third, both the computational complexity and the storage requirements are much lower compared to using the whole adjacency matrix. We develop a Python package DCD (The Python package is provided in [https://github.com/Ikerlz/dcd](https://github.com/Ikerlz/dcd).) to implement the distributed algorithm on a Spark system and establish theoretical properties with respect to the estimation accuracy and mis-clustering rates under the stochastic block model. Experiments on a variety of synthetic and empirical datasets are carried out to further illustrate the advantages of the methodology.

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

[Wu, S.](), [Li, Z.](https://ikerlz.github.io/) & [Zhu, X.](https://xueningzhu.github.io/), (2023) [A Distributed Community Detection Algorithm for Large Scale Networks Under Stochastic Block Models](https://www.sciencedirect.com/science/article/abs/pii/S0167947323001056).
