## Packaged Python Algorithm for Predicting Video Games Earnings

----

The aim of this project is to serve as a simple example of implementing tensorflow model to be used with the [sagemaker-pipeline](https://github.com/MustafaWaheed91/sagemaker-pipeline) project.

### Prerequisites

1. Make sure to have setuptools library installed on python

2. Using version 3.6+ of python


### Running Model locally

```

git clone https://github.com/MustafaWaheed91/tf-gamesbiz.git

cd tf-gamesbiz

pip3 install -e .

python3 gamesbiz/train.py

```

### Running Model on Amazon SageMaker

Follow the instructions in [sagemaker-pipeline](https://github.com/MustafaWaheed91/sagemaker-pipeline) project to see how to use the model
in this package to run with SageMaker training pipeline.

----

### Built primarily with

* [Tensorflow](https://www.tensorflow.org/) - Open Source Machine Learning framework


### Authors

* **Mustafa Waheed** - *Data Scientist* 
