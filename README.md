# `dcd`: a distributed spectral clustering package 

## What is it?
`dcd` is a Python package that provides a **Distributed Community Detection Algorithm** for network data based on **Spark**.

## Where to get it
The source code is currently hosted on GitHub at: https://github.com/Ikerlz/dcd

Binary installers for the latest released version are available at the [Python package index]().

```py
pip install dcd
```


## Dependencies

- `Spark >= 2.3.1`
- `Python >= 3.7.0`
  - `pyarrow >= 0.15.0` Please read this [compatible issue with Spark 2.3.x or 2.4.x](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#compatibility-setting-for-pyarrow--0150-and-spark-23x-24x)
  - `numpy >= 1.16.3`
  - `findspark >= 1.3.0`

## Installation from sources

In the pandas directory (same one where you found this file after cloning the git repo), execute:

```py
python setup.py install
```

## License
[MIT License](https://github.com/Ikerlz/dcd/blob/master/LICENSE)

## References

[Li, Z.](http://lizhe.fun/), & [Wu, S.]() [Zhu, X.](https://xueningzhu.github.io/), (2020) Distributed Community Detection for Large Scale Networks Using Stochastic Block Model. [_Working Paper_]().