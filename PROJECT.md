# Project objectives
1. Design, implement, debug, evaluate, and benchmark deep convolutional neural network architectures (and its variants) for a dataset created and curated by yourself with at least a 1000 images (NO pre-cleaned datasets, but you can collect images from the internet).  
    Professor Andrew Ng talks about the value of working on your own datadet in [this](https://youtu.be/1k37OcjH7BM) podcast.
1. Compare performance with a basic fully convolutional neural network model with a few filters and one or two layers.
1. Study the effect of data augmentation, regularization, and transfer learning.

# Project expectations
1. You will work on your projects individually (i.e. group submissions are not allowed).
1. All reports including the final report must be prepared using <a href="https://www.overleaf.com/">Overleaf</a>.
1. Each of you will review at least two reports of the peers in your group.

# Tip for preparing notebooks
1. Read the [Ten Simple Rules for Reproducible Research in Jupyter Notebooks](https://arxiv.org/pdf/1810.08055.pdf)

# Project phases

## 1. Data preparation
1. Collect the images or take pictures  
   ~~If you are extracting images from a video make sure that the images are not too easy to predict~~
1. Crop/resize them all to same dimensions (height = width)
1. Visualize sample images and discuss the distribution of output labels
1. Discuss data normalization

**What to submit?**  
a) A link to your Colab notebook (make sure that anyone with the link can view)  
b) A PDF report describing your findings  

## 2. Build an overfitting model
1. Using all the data (i.e. without splitting) obtain close to 100% accuracy. Build as large model as you need (with many filters and many layers). Here is an example:
   ```python
   model = Sequential()
   model.add( Conv2D( 64, ( 3, 3 ), activation = 'relu', input_shape = xtrain[0, :, :, :].shape ) )
   model.add( MaxPool2D(4, 4) )
   model.add( Conv2D( 32, ( 3, 3 ), activation = 'relu' ) )
   model.add( MaxPool2D(4, 4) )
   model.add( Conv2D( 16, ( 3, 3 ), activation = 'relu' ) )
   model.add( Flatten() )
   model.add( Dense( 10, activation = 'relu' ) )
   model.add( Dense( 10, activation = 'softmax' ) )
   ```
1. How does the performance (accuracy, precision, recall, etc.) change when the number of filters and layers are increased/decreased?
1. If you provide the output as the input (as an additional channel) what is the smallest architecture (minimum number of layers and filters) you need to overfit the data?
   ```python
   # Example of how to use output labels as additional input channel
   import numpy as np
   N = len(xtrain[:, 0, 0, 0])
   L = len(xtrain[0, :, 0, 0])
   xtrain_with_outputlabels = np.zeros((N, L, L, 2))
   for i in range(len(xtrain)):
      existing = xtrain[i, :, :, :]
      newchannel = np.full((L, L), ytrain_original[i]).reshape(L, L, 1)
      x = np.concatenate((existing, newchannel), axis = -1)
      print(existing.shape, newchannel.shape, x.shape)
      xtrain_with_outputlabels[i] = x
      break
    ```
   If you are using data generators, you can do something like the following to obtain your `xtrain` and `ytrain_original`:
   ```python
   # Empty placeholders for 1000 RGB images and their labels
   mydatax = np.zeros(1000, 256, 256, 3)
   mydatay = np.zeros(1000, 1)
   # Read everything from your generator
   for i in range(1000):
      x, y = your_generator()
      mydatax[i] = x
      mydatay[i] = y
   ```
1. Plot your learning curves and include them in your report

**What to submit?**  
a) A link to your Colab notebook (make sure that anyone with the link can view)  
b) A PDF report describing your findings  

## 3. Split and evaluate on test set
1. Split your data into training, development, and test set
1. Train your model using the training set, 'Earlystop' using the validation set, and evaluate on the test set
1. Study the performance when the number of filters and layers are increased/changed
1. Plot your learning curves and include them in your report

**What to submit?**  
a) A link to your Colab notebook (make sure that anyone with the link can view)  
b) A PDF report describing your findings  

## 4. Effects of augmentation
1. With the best model obtained from the previous step, apply various techniques of data augmentation (Image generators) and study the improvement in accuracy
1. Plot your learning curves and include them in your report

**What to submit?**  
a) A link to your Colab notebook (make sure that anyone with the link can view)  
b) A PDF report describing your findings  

## 5. Effects of regularization
1. With the best model obtained from the previous step, apply various techniques of regularization (Batchnormalization, Dropout, L2 regularization, etc.) and study the improvement in accuracy
1. Plot your learning curves and include them in your report

**What to submit?**  
a) A link to your Colab notebook (make sure that anyone with the link can view)  
b) A PDF report describing your findings  

## 6. Use pretrained models and recent architectures 
1. Use pretrained models such as VGG16 or ResNet50 and retrain using your dataset.
1. Use recent architectures such as ResNet, DenseNet, or NASNet to train a model and study the improvement in accuracy
1. Plot your learning curves and include them in your report

**What to submit?**  
a) A link to your Colab notebook (make sure that anyone with the link can view)  
b) A PDF report describing your findings  

## 7. Report
**What to submit?**   
1. A copy of your final report    
    * Your report must not be very long; 10/12 pages at most.
    * All tables and figures must be numbered and captioned/labelled.
    * Don't fill an entire page with a picture or have pictures hanging outside of the page borders.
    * It is encouraged but not required to you host your project (and report) at Github.   
1. A link to your final Notebook

## ~~8. Poster presentation (final exam)~~
* Prepare a poster highlighting the problem, diagrams, your results, plots, etc.
* Poster and final report will be evaluated based on all the criteria above.
* Report and poster presentation grades will be average of (a) grades by the course instructor (b) peer grades, and (c) grades by external visitor/s from industry
* See sample-reports folder and sample-posters folder for samples

**What to submit?**  
a) A copy of your final report  
b) A link to your final Notebook

**What to bring?**
a) A printed copy of your final report  
b) A link to your final Notebook
c) Your poster  
d) A laptop for demonstration (if needed)  

# Example Project
The goal in this project is to develop a convolutional neural network model that can identify my mood looking at a picture of my face. Here are the steps involved:
1. Take 1000 pictures of my face in various settings - smiling, laughing, sad, crying, and neutral - 200 images each. Then, label each of these pictures.
1. Crop images to 256 x 256 dimensions.
1. Write a Python matplotlib code to visualize all the 1000 images. 
1. Randomly spit the data into - 600 pictures for training, 200 for validation, and 200 for testing. 
1. Build a single layer CNN model with 64 filters, train the model, and evaluate the model on the test set. It is worth noting that the 5-class accuracy of a random classifier is 20% (baseline for the project).
1. Apply data augmentation techniques and regularization techniques to improve performance.
1. Build and test newer architectures such as ResNets and pre-trained models such as VGG-16.

<img src="syllabus/mood-classification-project.png" align="middle" width="800"/>


## Previous semester student projects
* [Mood Detection](https://github.com/jnkx9c/DL_Project) - by Jeff Killgore
* [Categorizing Equus Members with Deep Learning](https://github.com/mykon23/DL-2020Project/blob/master/Categorizing_Equus_Members_with_Deep_Learning.pdf) - by Miguel Corona
* [Finger Digit Classification](https://github.com/kvong/DeepLearning-FingerDigitClassification) - by Khanh Vong

