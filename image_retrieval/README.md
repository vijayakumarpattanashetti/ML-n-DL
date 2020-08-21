# Image Search


## Problem
Given an input image, output N similar images from the unlabeled dataset.


## Solution/Algorithm 
This is an Image Retrieval problem i.e., finding the images similar to the query image from the unlabeled dataset. We know that dataset contains ~5K 512x512 unlabeled RGB images. This makes computation cost too high, so we have to move on to dimensionality reduction. Also, we have to choose unsupervised learning algorithm as we have unlabeled data. This directs us to choose Autoencoders, which are the neural networks basically made up of encoder, decoder layers. Further, we get encoded image data of reduced dimensions making it quite better with respect to computation. We then apply, k Nearest Neighbors(kNN) algorithm to find similar images. So, we have split the problem into two sub problems viz., Image Dimensionality Reduction, and Finding Similarity.

We know that Convolutional Neural Networks(CNNs) are best suited neural architecture for Computer Vision(Image related) applications. Similarly, we employ Convolutional Autoencoders(ConvAE) to deal with our problem, as it is all about image.


## Tech Stack
Google Colaboratory, OpenCV, TF-Keras, Scikit-learn


## Resources
Dataset Set: https://drive.google.com/file/d/1VT-8w1rTT2GCE5IE5zFJPMzv7bqca-Ri/view?usp=sharing
Colab: https://drive.google.com/file/d/13_oKNkRp5Ru1HdOHSh8wMUp-C9BKIn0u/view?usp=sharing


## Neural Network Architecture – Convolutional Autoencoder(ConvAE)
<img src="https://github.com/vijayakumarpattanashetti/ML-n-DL/blob/master/image_retrieval/readme_images/arch.PNG">

## Flow chart
<img src="https://github.com/vijayakumarpattanashetti/ML-n-DL/blob/master/image_retrieval/readme_images/fd.PNG">
