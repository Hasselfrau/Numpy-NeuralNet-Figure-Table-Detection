# Logistic Regression model with a Neural Network mindset using only numpy.

In this project presented implementation of a Logistic Regression model with a Neural Network mindset using only numpy.
We are trying to solve two problems:
1. Predicting the Presence of Figures on a Document Page
2. Predicting the Presence of Tables on a Document Page

## Overview of the Problem set ##

**Problem Statement**:\
[taub_dataset.h5](https://www.kaggle.com/datasets/mmgoncharov/taub-center-7-open-source-reports) should be downloaded and installed here: /Users/USER/Projects/Numpy-NeuralNet-Figure-Table-Detection/data\
We have a dataset of 7 reports by [Taub Center](https://www.taubcenter.org.il/) [taub_dataset.h5](https://www.kaggle.com/datasets/mmgoncharov/taub-center-7-open-source-reports) contains:
  - A set of 257 pages labeled as containing figure (y_fig=1) or not containing (y_fig=0)
  - And labeled as containing table (y_table=1) or not containing (y_table=0)
  - Each image of page is of shape (681, 440, 3) where 3 is for the 3 channels (RGB).

We will build a simple image-recognition algorithm that can correctly classify pages as "with figure" or "without figure".

## General Architecture of the learning algorithm ##

We will build a Logistic Regression, using a Neural Network mindset.

**Mathematical expression of the algorithm**:

For one example:\
$$x^{(i)}$$:
$$z^{(i)} = w^T x^{(i)} + b$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})$$
$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})$$

The cost is then computed by summing over all training examples:\
$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})$$

## Forward and Backward propagation ##

Now that our parameters are initialized, we can do the "forward" and "backward" propagation steps for learning the parameters.

### Propagate ###

Forward Propagation:
- We get X
- We compute $$A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$$
- We calculate the cost function: $$J = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$$

Here are the two formulas we will be using:

$$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T$$
$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})$$

## Optimization ##
- We have initialized our parameters.
- We are also able to compute a cost function and its gradient.
- Now, we want to update the parameters using gradient descent.

The goal is to learn $$w$$ and $$b$$ by minimizing the cost function $$J$$. For a parameter $$\theta$$, the update rule is $$ \theta = \theta - \alpha \text{ } d\theta$$, where $$\alpha$$ is the learning rate.

## Prediction ##
The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X.
To do it we need:
1. Calculate $$\hat{Y} = A = \sigma(w^T X + b)$$
2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector `Y_prediction`.

## Merging all functions into a model ##

We will now see how the overall model is structured by putting together all the building blocks (functions implemented in the previous parts) together, in the right order.

To implement the model function we will se the following notation:
    - Y_prediction_test for our predictions on the test set
    - Y_prediction_train for our predictions on the train set
    - parameters, grads, costs for the outputs of optimize()

## Predicting the Presence of Figures on a Document Page ##

### Choosing learning rate ###

In order for Gradient Descent to work we must choose the learning rate wisely. The learning rate $$\alpha$$  determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.

Let's compare the learning curve of our model with several choices of learning rates.

**Result**:
- Different learning rates give different costs and thus different predictions results.
- If the learning rate is too large (0.01), the cost may blow up and raise RuntimeWarnings:
    - overflow encountered in exp; s = 1 / (1+ np.exp(-z))
    - divide by zero encountered in log; cost = - np.sum(Y * np.log(A) + (1-Y)*np.log(1-A)) / m
    - invalid value encountered in multiply; cost = - np.sum(Y * np.log(A) + (1-Y)*np.log(1-A)) / m

- Learning rate 0.0001 is also too large, we can se that the cost oscillating up and down.
- A lower cost doesn't mean a better model. We have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.

### Training the model ###
- num_iterations=5000
- learning_rate=0.00003

Cost after iteration 0: 0.693147\
Cost after iteration 100: 2.826988\
Cost after iteration 200: 2.646462\
...\
Cost after iteration 1900: 0.537395\
Cost after iteration 2000: 0.452311\
Cost after iteration 2100: 0.128279\
Cost after iteration 2200: 0.114696\
...\
Cost after iteration 4800: 0.065874\
Cost after iteration 4900: 0.064887\
train accuracy: 100.0 %\
test accuracy: 77.35849056603773 %

**Comment**: Training accuracy is close to 100%. This is a good sanity check: our model is working and has high enough capacity to fit the training data. Test accuracy is 77%.

It is actually not bad for this simple model, given the small dataset we used and that logistic regression is a linear classifier.
Also, we can see that the model is clearly overfitting the training data.
There is an elbow on 2100 iterations on the Cost function / gradients plot (Cost after iteration 2000: 0.452311, after iteration 2100: 0.128279). Let's decrease the number of iterations. We might see that the training set accuracy goes up, but the test set accuracy goes down. This is called overfitting.

**So we chose**:
- num_iterations=2100
- learning_rate=0.00003

And got results:

              precision    recall  f1-score   support

           0       0.80      0.65      0.71        31
           1       0.61      0.77      0.68        22

    accuracy                           0.70        53

**Precision**\
For class 0, the precision is 0.8, indicating that among all instances predicted as class 0, 80% were correctly classified.\
For class 1, the precision is 0.61, meaning that 61% of instances predicted as class 1 were indeed class 1.


**Recall**\
For class 0, the recall is 0.65, indicating that the model correctly identified 65% instances of class 0.\
For class 1, the recall is 0.77, suggesting that the model missed 23% instances of class 1.

## Predicting the Presence of Tables on a Document Page ##

### Training the model ###
With parameters
- num_iterations=2100
- learning_rate=0.00003

Cost after iteration 0: 0.693147\
Cost after iteration 100: 0.324088\
Cost after iteration 200: 0.280916\
...\
Cost after iteration 1900: 0.078135\
Cost after iteration 2000: 0.074816\
train accuracy: 97.54901960784314 %\
test accuracy: 92.45283018867924 %


## Quality ##

              precision    recall  f1-score   support

           0       0.92      1.00      0.96        46
           1       1.00      0.43      0.60         7

    accuracy                           0.92        53

**Precision**

For class 0, the precision is 0.92, indicating that among all instances predicted as class 0, 92% were correctly classified.\
For class 1, the precision is 1, meaning that all instances predicted as class 1 were indeed class 1.

**Recall**

For class 0, the recall is 1.00, indicating that the model correctly identified all instances of class 0.\
However, for class 1, the recall is 0.43, suggesting that the model missed some instances of class 1.
