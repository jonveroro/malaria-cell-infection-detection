# malaria-cell-infection-detection

Detection of Malaria Infected cell images using different machine learning techniques

## Main Prerequisites

* Keras
* Tensorflow
* Light GBM
* XGBoost

## Usage
* Download image data on Kaggle Link Below - Extract data a on ```data/cell-images/Parasitized``` and ```data/cell-images/Uninfected```
* Creating a dataset:
    ```python dataset_creator -l 10000```
    ```-l``` - size of the dataset to be created
* Training a model
    ```python model.py -m lgbm```
    ```-m``` - type of training method to be used
     ```lgbm``` - Light Gradient Boosting
     ```xgbm``` - Xtreme Gradient Boosting
     ```cnn``` - Convolutional Neural Network
* Evaluation of models
  ```python prediction.py -l 100```
  ```-l``` - number of samples to be evaluated.

## Feature Engineering
  For the features used. I used a biult in algorithm on opencv called KAZE algorithm. This algorithm is used to extract features of images and compare it with the features of other images. Every images is composed of 2047 numerical features generated by KAZE algorithm. 

## Main Observations and Hypothesis
  Based on observation and multiple testings Gradient boosting methods are mostly effective on datasets with very large features. However the CNN method performed badly. During the training phase XGBoost Model and Light GBM model can fastly segregate the prediction values from nearly 0 and 1 upon increasing the number of boost rounds.

## Test Results

  * Light GBM Model Result

  Correct Predictions 85/100..
  0.85 Prediction Accuracy.
  0.018144053455026933 Uninfected prediction average.
  0.7884970210986213 Infected prediction average

  Correct Predictions 175/200..
  0.875 Prediction Accuracy.
  0.036188139453329375 Uninfected prediction average.
  0.7998932429236194 Infected prediction average

  Correct Predictions 900/1000..
  0.9 Prediction Accuracy.
  0.030278600837203615 Uninfected prediction average.
  0.8373384061660107 Infected prediction average


  * XGBoost Model Result

  Correct Predictions 86/100..
  0.86 Prediction Accuracy.
  0.04328362 Uninfected prediction average.
  0.8173453 Infected prediction average

  Correct Predictions 169/200..
  0.845 Prediction Accuracy.
  0.04755407 Uninfected prediction average.
  0.7872302 Infected prediction average

  Correct Predictions 894/1000..
  0.894 Prediction Accuracy.
  0.04033276 Uninfected prediction average.
  0.83465785 Infected prediction average



  * Keras CNN Result

  Correct Predictions 61/100..
  0.61 Prediction Accuracy.
  0.2773743637327878 Uninfected prediction average.
  0.3567644839591168 Infected prediction average

  Correct Predictions 122/200..
  0.61 Prediction Accuracy.
  0.284440261252383 Uninfected prediction average.
  0.3584368543804817 Infected prediction average

  Correct Predictions 641/1000..
  0.641 Prediction Accuracy.
  0.2975646531811618 Uninfected prediction average.
  0.37553351513664 Infected prediction average

## Improvements
  * Apply 2D CNN
  * Improve Hyperparameters on all models
  
## Built With

* [Keras](https://github.com/keras-team/keras) - Deep Learning Library
* [XGBoost](https://github.com/dmlc/xgboost) - Gradient Boosting Library
* [Light GBM](https://github.com/Microsoft/LightGBM) - - Gradient Boosting Library
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) - Machine Learning Library
* [Open-CV](https://github.com/opencv/opencv) - Computer Vision Library

## Authors

* **Jonver Oro** - *Initial work* - [jonveroro](https://github.com/jonveroro)

## Dataset
* (Malaria Cell Images Dataset)(https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)

## References
 * (Image feature Extraction using KAZE)(https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774)
 * (XGBoost Documentation)(https://xgboost.readthedocs.io/en/latest/)
 * (Light GBM Documentation)(https://lightgbm.readthedocs.io/en/latest/)

