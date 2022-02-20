# Setting Up an Image Classification ML Workflow with AWS

In this project, we will be training and deploying a classification model that attempts to identify motorcycle and bike images. We will be using various services in AWS, including:
- Sagemaker
- S3
- Lambda
- Step Functions

For this project, we will be using this [images dataset](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz). While the set contains a large variety of images, we will be looking specifically at those of bikes and motorcycles.

## Project Specifications
The notebook and endpoint were run in an AWS `ml.t3.medium` instance, with the notebook additionally using `Python 3 (MXNet 1.8 Python 3.7 CPU Optimized)`.

## Key Files
* `starter.ipynb` - The main notebook used to run commands and interact with SageMaker.
* `README.md` - Contains key information about set up and post-project analysis.
* `lambda.py` - Python file containing handler functions in AWS Lambda stages.
* `step_function_workflow.json` - JSON containing an AWS Step Functions configuration for our workflow.

## Project Set Up and Installation
1. Enter AWS through the gateway in the course and open SageMaker Studio.
2. Download the starter notebook file.
3. Proceed by running the cells once in consecutive order. For the `lambda.py` and `step_function_workflow.json` files, you will have to upload or paste them directly into the service interfaces.
4. If the images are not rendering, or you need a closer look, you can check out all of them in the root folder.

## Sources:
* Template code and dataset images from Udacity's AWS course
* Screenshots taken from AWS
