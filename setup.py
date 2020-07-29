import os
from setuptools import setup


def read(file):
    return open(os.path.join(os.path.dirname(__file__), file)).read()


setup(name='dcd',
      version='0.01',
      description='Distributed Community Detection Algorithm',
      keywords='spark, spark-ml, pyspark, mapreduce',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='https://github.com/Ikerlz/SparkSimulation',
      author='Zhe Li',
      author_email='ikerlizhe@gmail.com',
      license='MIT',
      packages=['dcd'],
      install_requires=[
          'pyspark >= 2.3.1',
          'numpy   >= 1.16.3',
          'pandas  >= 0.23.4',
          'findspark >= 1.3.0',
      ],
      zip_safe=False,
      python_requires='>=3.7',
)