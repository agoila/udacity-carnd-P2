# Traffic Sign Recognition

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/class_distribution.png "Visualization" 
[image2]: ./examples/sample_images.png "Sample Images from the training dataset"
[image3]: ./examples/min_images.png "Classes with min. examples in training dataset"
[image4]: ./examples/beforeandafter_gray.png "Grayscaling"
[image5]: ./examples/gray_clahe.png "Applying CLAHE"
[image6]: ./examples/tensorboard_graph.png "TensorBoard Graph"
[image7]: ./examples/tb_accuracy.png "Training Accuracy - TensorBoard"
[image8]: ./examples/tb_loss.png "Loss - TensorBoard"
[image9]: ./examples/tes.PNG "Test images from the web"
[image10]: ./examples/visualizations.PNG "Visualizing the Neural Network"

## Rubric Points
I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

Here is a link to my [project code](https://github.com/agoila/udacity-carnd-P2/blob/master/LeNet_Traffic_Sign_Classifier.ipynb)

### Dataset Summary & Exploration
#### Dataset summary
I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### Exploratory visualization
Here is an exploratory visualization of the data set. It is a bar chart showing the class distribution in the dataset.

![alt text][image1]

Also shown below are some sample images with their respective classes.

![alt text][image2]

Here are some classes that have the minimum number of examples in the training dataset (as observed from the bar chart above).

![alt text][image3]

### Design and Test a Model Architecture

#### Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 
As a first step, I decided to convert the images to grayscale because instead of learning to classify traffic signs based on color in addition to the classes, it's easier for the model to learn from grayscaled images - much less work and if a new sign comes up in a color that the model has never seen before, it's going to be a lot easier for the model that ignores the color to begin with to classify that sign. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

As a second step, I applied CLAHE to the grayscaled image data following instructions [here](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html). This was done to improve the contrast of the images. Look at how it stretches the histogram of a sample training image. 

![alt text][image5]

As the last step, I normalized the image data so that the data has zero mean and unit variance.


#### Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Pre-processed image (grayscale, CLAHE, normalized)   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| output depth = 120        									|
| Dropout 	| Keep probability = 0.5        									|
| Fully connected		| output depth = 84        									|
| Dropout 	| Keep probability = 0.5        									|
| Fully connected		| Logits, output depth = 43 (no. of classes)        									|
| Softmax				|     									|
 
Here's the graph visualization in TensorBoard:

![alt text][image6]

#### Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a learning rate of 0.001, batch size of 128 and trained it for 25 epochs. For an effective visualization of the training process, I also incorporated TensorBoard, and added histogram summaries and name scopes. 

#### Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 94.6% (or 0.946)
* validation set accuracy of 95.1% (or 0.951) 
* test set accuracy of 93.0% (or 0.930)

Here is a plot of training accuracy and loss, as visualized in TensorBoard:

![alt text][image7] ![alt text][image8]

In this project, I chose the LeNet architecture as the starting point. It is a great starting point for building a network to classify traffic signs, and is a simple architecture to begin with.

The initial LeNet architecture was setup for MNIST classification tasks, and thus needed to be changed. I updated the input and output layer depths to match the image size and the number of classes. 

With no dropout, I observed that the training accuracy was much higher than the validation accuracy (0.98 to 0.84), i.e the model was overfitting. Dropout layers were introduced after each fully connected layer as a regularization measure. 

These dropout layers are visible in the TensorBoard graph visualization above. Initial value was chosen to be 0.8 and after some finetuning, I achieved best accuracy with a 50% keep_probability on the dropout layers.

I trained for 10 epochs initially, but carefully adjusted it to 40 after observing the loss performance in TensorBoard.
 
### Test a Model on New Images

#### Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image9]

The first image might be difficult to classify because it belongs to a class (class 0) that has the minimum number of examples in the training/validation/test set. In short, the model doesn't see enough representative examples in the datasets and hence it hasn't learned very well on this type of image. 

#### Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)      		| General Caution   									| 
| Dangerous curve to the left     			| Dangerous curve to the left 										|
| Go straight or left					| Go straight or left											|
| Yield	      		| Yield					 				|
| End of all speed and passing limits			| End of all speed and passing limits      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93%. 

I had to manually crop the test images from the web and then fed the image in. This is because the model is not trained for object detection yet, and cropping out unnecessary noise from these test images improved my accuracy from 20% to 80%. 

#### Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the Ipython notebook.

For the first image, the model is a bit unsure that this is a 20km/h speed limit sign (probability of 0.077), and tries to predict a "General caution" sign with a probability of 0.79. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .79         			| General caution   									| 
| .077     				| Speed limit (20km/h) 										|
| .033					| Go straight or left											|
| .026	      			| Pedestrians					 				|
| .019				    | Speed limit (120km/h)      							|


For the second image, the model is a 100% sure that it is a "Dangerous curve to the left" sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| General caution   									| 
| 0     				| Speed limit (20km/h) 										|
| 0					| Go straight or left											|
| 0	      			| Pedestrians					 				|
| 0				    | Speed limit (120km/h)      							|

For the third image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9991         			| General caution   									| 
| .0008     				| Speed limit (20km/h) 										|
| 0					| Go straight or left											|
| 0	      			| Pedestrians					 				|
| 0				    | Speed limit (120km/h)      							|

For the fourth image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| General caution   									| 
| 0     				| Speed limit (20km/h) 										|
| 0					| Go straight or left											|
| 0	      			| Pedestrians					 				|
| 0				    | Speed limit (120km/h)      							|

And for the last image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9995         			| General caution   									| 
| .0005     				| Speed limit (20km/h) 										|
| 0					| Go straight or left											|
| 0	      			| Pedestrians					 				|
| 0				    | Speed limit (120km/h)      							|

As we can see, the model predicts right on all traffic signs except the first one. 

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here's the visualization of the trained network's feature maps for the first convolutional layer "conv1" for all the web test images:

![alt text][image10]
