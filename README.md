## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

My project code is included in this repository (Traffic_Sign_Classifier.ipynb).
An html file containing the code along with output for a complete run is also included
(Traffic_Sign_Classifier.html).

I installed TensorFlow with GPU support on my laptop, which has an Nvidia GPU.
TensorFlow with GPU support was observed to train my network 
roughly 9X faster than the CPU-only version.  
Without GPU support, the lab would have been infeasibly time-consuming.

---
### Writeup / README

My project code is included in this repository (Traffic_Sign_Classifier.ipynb).
An html file containing the code along with output for a complete run is also included
(Traffic_Sign_Classifier.html).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the len() function to determine size of training, validation, and test
sets, and the shape attribute to determine individual image shape.

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Please refer to heading **Include an exploration visualization of the dataset** in the html output.
A sample image from the training set is shown along with its label. 
Also, a histogram of the number of images of each type in each set is shown for all three sets. 
Some sign types are underrepresented, although the distribution of sign types in the training,
validation, and test sets is surprisingly similar.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did not consider converting to grayscale because it removes information before the network even
has a chance to play with it.

I normalized the data as follows, converting to 32-bit floating point in the process:
```python
f128 = np.float32(128)
X_train_unaugmented = (X_train_uint8.astype(np.float32)-f128)/f128
```

I played with generating additional data as well.  I wrote a function to take an image from the
normalized test set, rotate it by a small amount using scipy.ndimage.interpolation.rotate(),
and add a small amount of random noise using np.random.normal.

I wrapped this function in a loop that appended data to the training set such that at least 1000 
instances of each label were represented.  Labels to augment, and the number of augmented instances
to add to each label, were chosen using the histogram of each label computed earlier.
For a given label, each augmented image was added by first selected a random image from the original
(unaugmented) data, then applying the rotation+random noise function to it.

The total size of the original+augmented training set was precomputed, and storage preallocated,
to avoid calling append() in an inner loop.  

Please refer to heading **Add augmented images such that each sign type has at least 1000 examples**
in the html output or jupyter notebook for more information.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My network has essentially the same structure as LeNet from the lab.  The only differences were 
the following:

First, I modified the first layer to accept depth-3 (color) images instead of depth-1 images.  Before:
```python
conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
```
After:
```python
conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
```
that I modified the output layer

I also added a dropout layer after each activation (relu) layer, for example:
```python
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.dropout(conv1, keep_prob)
```

The myLeNet function accepted keep_prob as an additional parameter.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


I trained with a dropout keep_prob of 0.75, but took care to use keep_prob of 1.0 for the validation
step.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The cell "Output Top 5 Softmax Probabilities For Each Image Found on the Web" 
of the jupyter notebook or html output contains my 
code for outputting softmax probabilities for each image from the web.
The top five softmax probabilities for each image are listed, along with bar charts.

In all cases the guess is correct. For all but the "bicyle crossing", the softmax probability is >= 96%.

Please refer to the jupyter notebook or html output for details.  
