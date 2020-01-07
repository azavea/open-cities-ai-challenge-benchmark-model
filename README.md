# Open Cities AI Challenge benchmark model
Benchmark model for DrivenData Open Cities AI Challenge

This repository is designed to be a resource for contestants of [DrivenData's Open Cities AI Challenge](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/page/150/). The objective of this challenge is to develop a machine learning model that can segment out buildings from aerial imagery of several African cities. The code and documentation contained here demonstrate how to train a model and make predictions on the competition data using [Raster Vision](https://rastervision.io/).

## Raster Vision
Raster Vision (rv) is an python framework that makes it easy to implement existing backend deep learning frameworks for computer vision tasks that use satellite, aerial and other large format imagery. To learn more about what Raster Vision can do, check out [the docs](https://docs.rastervision.io/en/0.9/) or [this blog post](https://www.azavea.com/blog/2018/10/18/raster-vision-release/) announcing its release. If you are interested in tutorials on using Raster Vision outside the context of this competition, there are several examples in the [raster-vision-examples repo](https://github.com/azavea/raster-vision-examples).

## Instructions

An rv experiment can be run either locally or on [AWS Batch](https://aws.amazon.com/batch/) but to train a deep learning model you will need access to GPUs. This example demonstrates how to train a model on AWS Batch. Using batch is beneficial not only because we gain access to GPUs but it also allows us to split some of the tasks across many instances and speed up the process. If you would like to reproduce this example on AWS Batch, begin by following the configuration instructions within the [Raster Vision AWS repo](https://github.com/azavea/raster-vision-aws).

1. *Download test STAC*
PyStac is able to access STAC catalogs on s3 but it is much faster to read them locally. It will not be a problem to read the training STAC directly from the DrivenData hosted s3 bucket but given that there are so many items in the test set (over 11,000), we recommend that you download the STAC and read it locally. It will end up being much faster than trying to read from s3 each time. 

- Download the test data from [the data download page](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/data/) and unpack it into a 'test' directory within the data folder in this repo

2. *Build and publish docker resources*
This experiment is designed to run within a pre-configured docker environment. You can build, publish and run the docker resources using scripts within the `docker` directory.

Build the docker image:
```
./docker/build
```
This will create a local docker image called 'raster-vision-wb-africa' that includes the code within the `benchmark` module, which you will need to run this experiment. 

If you would like to run the workflow on remotely you will need to publish the image to AWS Batch. 
- Edit `docker/publish_image` to reference the ECR cpu and gpu repos that you created during the [Raster Vision AWS setuo](https://github.com/azavea/raster-vision-aws#raster-vision-aws-batch-runner-setup).
Publish your docker image to both gpu and cpu repos:
```
./docker/publish_image
```
Both images will be tagged with `world-bank-challenge`. When/if you make changes to the aux commands you will need to repeat this process (i.e. rebuild the docker image and publish the updated version to ecr).

Run the docker container with the run script:
```
./docker/run --aws
```
The `--aws` flag forwards your AWS credentials which you will need if you plan to run any jobs remotely or access data on s3. However if you store your data and train the model locally you can run the script without it.