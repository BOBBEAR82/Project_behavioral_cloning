# **Behavioral Cloning** 

## Writeup



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/need_more_steering.jpg "need_more_steering"
[image2]: ./examples/bridge_area.jpg "bridge_area"
[image3]: ./examples/close_to_lake.jpg "close_to_lake"
[image4]: ./examples/recover_from_side.jpg "recover_from_side"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (didn't make any changes)
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of convolution neural network with 5 convolutional layers and 3 fully connected layers. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers (20% dropout) following each of the 5 convolution layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. By monitoring the loss from the training and validation, the epoch value was modified to make sure the loss are closed between training and validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I took 3 counterclockwise laps and 2 clockwise lap to make sure I can collect enough data. Then I collect more data for places where more steering is needed, as well as the bridge area.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the example in the lesson: 5 convolution layers followed by 3 full connected layers. I thought this model might be appropriate because there should be deep enough to capture all different features for self driving. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. After training with this, I found the validation loss started increasing after 5 epochs. That is the reason why I first set the training epoch as 5. With this trained model running in the simulator, I found that it worked pretty well in the counter clockwise lap, except one place right after crossing the bridge where the car hit the right side line a little bit before going back to center of the lane. But didn't worked very well in the clocewise lap. The car went out of the lane at the place closed to the lake. 

Then I made some changes to the model by
* adding dropout layers after each convolution layer to combat over fitting
* adding RELU activation to the first and second full connected layer to achieve nonlinear operation.

After this change, I trained the model again with the same data. This time both training loss and validation loss continued dropping with epoch of 10. So I kept the training epoch as 10. By running the model in the simulator, I can see that the performance was improved. The car can drive around track one both counter clockwise and clockwise, without touching either left or right line. 

#### 2. Final Model Architecture

The final model architectureconsisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 image   					| 
| Normalization		    |   					| 
| Cropping		        | Cropping the image by 70 pixels on top and 25 pixels at the bottom 					| 
| Convolution 1     	| 2x2 stride, depth is 24 |
| Activation			|RELU												|
| Dropout			    |20%												|
| Convolution 2     	|  2x2 stride, depth is 36  	|
| Activation			|RELU												|
| Dropout			    |20%												|
| Convolution 3     	|  2x2 stride, depth is 48  	|
| Activation			|RELU												|
| Dropout			    |20%												|
| Convolution 4     	|  1x1 stride, depth is 64  	|
| Activation			|RELU												|
| Dropout			    |20%												|
| Convolution 5     	|  1x1 stride, depth is 64  	|
| Activation			|RELU												|
| Dropout			    |20%												|
| Flatten   	      	|  					|
| Fully connected 1		| Outputs 100  									|
| Activation			|RELU												|
| Fully connected 2		| Outputs 50  									|
| Activation			|RELU												|
| Fully connected 3		| Outputs 1 									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3 laps on track one using center lane driving. 

I then recorded 2 laps on track one, still using center lane driving, but in counter clock direction.

After training the model using this data, I found it was not working pretty well at some places where
* need more steering
* near the bridge area
* close to the lake

Then I recoreded more data for all these areas, as well as more recovering from side back to the center of the road.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would, as well as used data from all three cameras. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the loss from both training and validation, which are both dropping, more than 10 epochs will cause validation loss start increasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.

All the training was performed on AWS EC2, which is much faster than running on local laptop.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
