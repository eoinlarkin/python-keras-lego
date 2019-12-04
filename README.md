# python_keras_lego

## Overview

This code implements an image classification model to identify LEGO pieces utilising the Keras deep-learning framework. 
This project has been inspired by an article on the IEE Spectrum website [(link)](https://spectrum.ieee.org/geek-life/hands-on/how-i-built-an-ai-to-sort-2-tons-of-lego-pieces) which detailed a LEGO sorting machine that used a machine learning model to classify individual LEGO pieces. 

## Dataset
For the purposes of the initial version of the model, images of Lego bricks have been sourced from a public dataset available from the statistical learning website Kaggle.com [(link)](https://www.kaggle.com/joosthazelzet/lego-brick-images).
The dataset contains 400 images for different types of LEGO pieces. The Lego bricks have been created using the Blender 3D Modelling application. Each individual image displays the bricks from a different angle.

For the purposes of training and testing the initial version of the model, I have decided to focus on four distinct brick types:
* 3022 Plate 2by2
* 3069 Flat Tile 1by2	
* 3040 Roof Tile 1by2by46deg
* 6632 Technic Lever 3M

![Raw Bricks](/_ref/raw_bricks.png)

## Modified Dataset
For the Keras framework, best practices indicates that it is useful to “simplify” the input images in order to obtain a more efficient model output and to reduce run times.
The raw input images from my dataset are of 200 pixels by 200 pixels with 3 dimensions (RGB colour space) . Initially I resized my images to 32 pixels by 32 pixels with 1 dimension of colour.
As we can see in the output image below, there is a degree of pixilation in our images, however the resulting transformed image seems to retain sufficient information to enable the fitting of the model. 

![Modifed Bricks](/_ref/modified_brick_images.png)

## Defining the Model & Hyper-Parameters
The model specification was as follows:
* A Convolution Neural Network (CNN) was used for the classification. This is considered a state-of-the art model for image detection and classification .
* A convolution and pooling step configured to the dimensions of the input data. A Rectified Linear Unit (ReLU) is used for the activation function, with the subsequent output having a filter size of 128.
* This convolution layer is duplicated, however the second layer includes a dropout of 25%. A dropout helps prevent overfit in our model; in essence, the dropout performs a form of averaging and prevents individual neurons from becoming dominat . The final output again uses a filter size of 128.
* The model output is then passed to a flattening layer
* Following this there is a dense layer. For this I have specified a filter size of 64, with the intention being to “compress” the output of the CNN.
* The final layer is a dense layer with an output filter of size 4 – this is equal to the number of categories of LEGO pieces that we are attempting to predict.
* The loss function of the model is sparse_categorical_crossentropy
* The optimizer is ADAM
* For determining the optimal model fit, the Keras model uses the “accuracy” condition. 

## Model Fit
Once the model has been defined, it was ran for 3 Epochs.
The model converges to an accuracy of 97% for the Training dataset and this is visible in the following graph, along with the corresponding loss function for the fit

![Model Accuracy](/_ref/model_accuracy.png)

It is also possible to examine the model accuracy by brick type :

![Accuracy by brick type](/_ref/model_accuracy_train_valid.png)

Interesting, while our overall accuracy is extremely high, there is a large divergence in accuracy between the training and validation datasets and also between individual bricks.
* The “6632 Technic lever” shows 100% accuracy for both the Train and Validation datasets
* Both the “3022 Plate” and the “3069 Flat Tile” show high levels of accuracy, both for the Train and Validation datasets, over 90% in both cases
* The model seems to be having issues with the “3040 Roof Tile”, particularly for the Validation dataset. This may indicate that the model is overfitting for the Training dataset.

