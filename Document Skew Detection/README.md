# Introduction

We all would have stumped on a problem where the documents that we have will be misaligned,
skewed and also could be wrapped. A lot of images scanners will ask us to rotate the images
by ourselves or ask us to choose the four points for modification of perspective.
But what is meant exactly by skew, it's any deviation of the image from that of the original
document which is not parallel to the horizontal or vertical.
Skew corrections remain one of the vital parts in Document processing.

Estimating and rectifying the orientation angle of any image is a pretty challenging task. Problem
of skewed documents degrade the performance of OCR and image analysis system so to detect
and correct of skew angle is important step of preprocessing
of document analysis.
In order to solve this problem we have various techniques to solve this problem like Projection
profile, Hough Transform, Nearest Neighbor connectivity, Fourier transform , and linear
Regression analysis and mathematical morphology. **But we are going to work on this problem
using Deep Learning approaches.**


# Data Description

The dataset from kaggle (https://www.kaggle.com/datasets/vishnunkumar/rdocuments?select=rdocuments), consists of a CSV file that contains the images "id" which is the image name and their rotation angle in **degrees**, with a file of length of 950 rows (no. of images)

![Sample Image](https://github.com/mostafasamii/Computer-Vision-Projects/tree/main/Document%20Skew%20Detection/repo_imgs/sample.png)

and a folder of Images "rdocuments" contains 950 images of skewed documents. and here you are a sample of the data

![Skewed Docs Image](https://github.com/mostafasamii/Computer-Vision-Projects/tree/main/Document%20Skew%20Detection/repo_imgs/skewed_docs.png)

and the distribution of the scaled angles in skewed images:

![Angle distribution Image](https://github.com/mostafasamii/Computer-Vision-Projects/tree/main/Document%20Skew%20Detection/repo_imgs/angles_dist.png)

Then I divided the dataset between 760 images for training and 190 for validation. And I got these results from training the model without the transfer learning

![Model History Image](https://github.com/mostafasamii/Computer-Vision-Projects/tree/main/Document%20Skew%20Detection/repo_imgs/model_history.png)


# Experiments

From an overview I am using CNN and building a single regression task. Ok how I did that
Firstly, I read the images in greyscale and resize them by 224, then transform the angle from
Degrees to Radian and apply StandardScaler to normalize the angles.
After that I started building an image generator with some data augmentations like rescaling the
images, applying zoom range or changing the brightness range to be (0.8 ~ 1.2).
Then I built the model as following

![Model Layers Image](https://github.com/mostafasamii/Computer-Vision-Projects/tree/main/Document%20Skew%20Detection/repo_imgs/model_layers.png)

I trained the model for 50 epochs with a batch size of 32. And the loss function for the CNN
model was Mean Absolute Error (MAE)
The results from the model using a test image is:

![Test Image](https://github.com/mostafasamii/Computer-Vision-Projects/tree/main/Document%20Skew%20Detection/repo_imgs/test_image.png)
![Resulting Image](https://github.com/mostafasamii/Computer-Vision-Projects/tree/main/Document%20Skew%20Detection/repo_imgs/resulting_image.png)


# Future Work

* Optimize accuracy
* Use transfer learning


# Tools
* Google Colab Notebook which is a data science platform allows you to combine executable code and rich text in a single document along with images
* Google Drive to store data online and read it in Colab
* Python version 2.7.12


# Overall Conclusions

The algorithm has been tested in the data of the input documents formats and has been found to detect the page orientation and existing skew successfully.


# External Resources
* https://medium.com/mlearning-ai/skew-correction-in-documents-using-deep-learning-8e19609107b6
* https://www.researchgate.net/publication/279450553_A_Review_of_Skew_Detection_Techniques_for_Document
* https://www.sciencedirect.com/science/article/pii/0923596594900094
* https://www.mdpi.com/2079-9292/9/1/55/htm
