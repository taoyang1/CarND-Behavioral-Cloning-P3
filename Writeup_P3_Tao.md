
# **Behavioral Cloning** 

## Tao Yang

### 05/14/2017

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_lane_driving.jpg "center lane driving"
[image3]: ./examples/recover_from_left.jpg "Recovery from left"
[image4]: ./examples/recover_from_right.jpg "Recovery from right"
[image6]: ./examples/loss_plot.png "loss plot as a function of epochs"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md and writeup_report.pdf summarizing the results
* video.mp4 which contains one laps of autonomous driving footages

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the NVIDIA architecture. 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Dropout is used for avoid overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105s).
The number of epochs is tested manually to make sure both the validation and training loss decreases monotonically. As a result, nb_epoch = 5 is chosen. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, adjusted left and right camera data, and a reverse lap data.

I also convert the original BGR data to RGB format so that it is consistent with driver.py.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build on an existing deep neural network.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because track 1 is relatively simple.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had both large mse. This suggests I'm underfitting.

To combat the underfitting, I adopted the NVIDIA model since it's more complex. Now it's better.

Then I add the normalization and cropping layer to better pre-process the data.

I also augmented the training data by taking the left and right camera data. A correction term is add/subtracted from the original steering angles. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collect more data focused on recovery from left and right and also a reverse lap driving data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 83-99) consisted of a convolution neural network with the following layers and layer sizes:

1. A data normalization layer.
2. A cropping layer where the top 55 pixels and bottom 25 pixels are dropped.
3. A conv2d layer with 5*5 filter sizes, stride of 2, depth of 24 and a RELU activation layer.
4. A conv2d layer with 5*5 filter sizes, stride of 2, depth of 36 and a RELU activation layer.
5. A conv2d layer with 5*5 filter sizes, stride of 2, depth of 48 and a RELU activation layer.
6. A dropout layer with rate = 0.4.
7. A conv2d layer with 3*3 filter sizes, stride of 1, depth of 64 and a RELU activation layer.
8. A dropout layer with rate = 0.4.
9. A conv2d layer with 3*3 filter sizes, stride of 1, depth of 64 and a RELU activation layer.
10. A dropout layer with rate = 0.4.
11. A fully connected layer with 100 outputs.
12. A dropout layer with rate = 0.3.
13. A fully connected layer with 50 outputs.
14. A fully connected layer with 10 outputs.
15. Final output with 1 continuous output being the steering angle.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 6 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from drift to the left or the right. These images show what a recovery looks like starting from the left or right.

**Recover from left**

![alt text][image3]

**Recover from right**

![alt text][image4]


I did not use track two in this project.

In order to be consistent with image format in drive.py, I convert the original BGR image format to RGB.

To augment the data sat, I include both data from the left camera and the right camera. Their corresponding angles are obtained by adding/substracting a constant correction (0.25) to the center angle.

I also collect a reverse lap driving data to compensate the left steering bias.

After the collection process, I had about ~55K of images. I then preprocessed this data by normalization and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was  chosen to be 5 by inspecting the loss plot below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![loss_plot][image6]

## Evaluation Video

The final testing video is in [this](https://github.com/taoyang1/CarND-Behavioral-Cloning-P3) git repository. Note one lap of autonomous driving is captured.


```python

```
