*Behavioral Cloning* 

---
**Udacity Self-Driving Car**
**Term1 P3: Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/model_mse_loss.png "Model MSE Loss"
[image3]: ./examples/center.jpg "Center Image"
[image4]: ./examples/left.jpg "Left Image"
[image5]: ./examples/right.jpg "Right Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
speed in drive.py is modified from 9 to 20 to make the car drive faster.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is similary to [Nvidia Network Architechure](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with 1 normailized layer, 5 convolutional layer, 4 funlly-connected layers, 3 dropout layer. (model.py lines 80-95) 

The input data (both training and validation) is normalized in the model using a Keras lambda layer (code line 81). Then this model uses cropping to trim image to only see section with road (code line 82).

#### 2. Attempts to reduce overfitting in the model

The model contains 3 dropout layers in order to reduce overfitting (model.py lines 86,88 and 90).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 29 and line 66). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 97).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of 2 laps of center lane driving, 2 laps of reverse direction driving, 1 lap focusing on driving smoothly around curves. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for driving a model architecture was to choose a suitable existing model and modify that model according to training process, validation loss, etc.

My first step was to use a convolution neural network model similar to the Nvidia auto drive team's Network .  I thought this model might be appropriate because they use it for training real car driving autonomously. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To avoid the overfitting, I modified the model by adding 3 dropout layers after 3 convolutional layers. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially at sharp turn. I believe this is more affected by training data rather than the model architecture. To improve the driving behavior in these cases, I tried to collect more data on turing around curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
Please check [run1.mp4](https://github.com/neo-cc/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4)

#### 2. Final Model Architecture

The final model architecture (model.py lines 80-95) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 				   					| 
| Cropping 		     	| 50/20 rows of pixels from top/bottom		 	|
| Convolution 	    	| 24 filters, 5x5 kernel, valid padding 		|
| RELU					|												|
| Convolution		    | 36 filters, 5x5 kernel, valid padding			|
| RELU					|												|
| Convolution	 		| 48 filters, 5x5 kernel, valid padding			|
| RELU					|												|
| Dropout 				| 0.5   										|
| Convolution 		    | 64 filters, 5x5 kernel, valid padding			|
| RELU					|												|
| Dropout 				| 0.5   										|
| Convolution 		    | 64 filters, 5x5 kernel, valid padding			|
| RELU					|												|
| Dropout 				| 0.5   										|
| Fully connected		| outputs 100 flat 								|
| Fully connected		| outputs 50 flat 								|
| Fully connected		| outputs 10 flat 								|
| Fully connected		| outputs 1 flat								|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Then I also recorded two laps on track one with reverse driving direction so the track seems completely new for the car. Here is an example image of center lane driving:

![center][image3]

I did not record the vehicle recovering from the left side and right sides of the road back to center. Instead, I used left camera and right camera images for training. +0.2 steering angle is added for left images and -2.0 for right images. These images teaches the car how to return to center even it's off the track.

![left][image4]
![right][image5]


To augment the data sat, I also flipped images and angles thinking that this would help to train the model because the model would think the flipped image as new training data. For example, here is an image that has then been flipped:

![normal][image6]
![flipped][image7]

After the collection process, I had 32970 number of data points. 
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. (I set epochs to 10 at first but validation loss is not decreasing after epoch 3) The validation loss I used an adam optimizer so that manually training the learning rate wasn't necessary.

![model_mse_loss][image1]

#### 4. Track Two test

I did not record any data from track two. When I used my trained model to test autonomously drive on track two, the car went off the road at the beginning and can never recover. Current trained model is only good for track one. 

