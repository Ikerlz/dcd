# `dcd`
**Distributed Community Detection** Algorithm implemented with Apache Spark

# System Requirements

- `Spark >= 2.3.1`
- `Python >= 3.7.0`
  - `pyarrow >= 0.15.0` Please read this [compatible issue with Spark 2.3.x or 2.4.x](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#compatibility-setting-for-pyarrow--0150-and-spark-23x-24x)
  - `numpy >= 1.16.3`
  - `findspark >= 1.3.0`

  See [`setup.py`](setup.py) for detailed requirements.

# Run the [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) code on the Spark platform
```sh
  ./bash/spark_dcd_run.sh
 ```
 or simply run

 ```py
   ./spark_dcd.py
 ```

# References

- [Li, Z.](http://lizhe.fun/), & [Wu, S.]() [Zhu, X.](https://xueningzhu.github.io/), (2020) Distributed Community Detection for Large Scale Networks Using Stochastic Block Model. [_Working Paper_]().
