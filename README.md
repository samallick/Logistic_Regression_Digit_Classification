# Logistic Regression - Recognising Hand-Drawn Digits



## Example Results:

Where the hand-drawn digits are in the centre and the program's guesses are in the top left corners.

![](/home/sam/downloads/results.png)

## Installing Dependencies, Compiling, Running and Modifying the Program:

##### Dependencies

- Written in C++
- Dependencies:
  - ​	[OpenCV 4.2](https://github.com/opencv/opencv)

Note that OpenCV is used here only for collecting pixel data from images, performing matrix operations, and displaying images (results). This program does not use external libraries for the classification problem itself.

##### Compiling

On Ubuntu, after installing OpenCV, I can compile C++ code with OpenCV headers included by using the form: ``g++ example.cpp -o example `pkg-config --cflags --libs opencv4` `` (note the backticks). 

However, I usually use CMake. A CMakeLists.txt file is included in each of the *learn* and *predict* programs. Change into the build directory and execute `cmake ..` then `make` then `./predict` (or `./learn`).

##### Running

`./learn` iteratively computes better parameters as it learns to guess our training examples (from **digits/training_data**) correctly. This process is called training. When it's done, those parameters are output to the file **digits/learn/build/classifier.txt**

`./predict` imports the parameters from **classifier.txt** then predicts the values of the hand-drawn digits in **digits/test_data**

The *learn* program has already been run and has produced a set of parameters. The *predict* program is ready to run.

##### Modification

The training data consists of 1000 images (100 hand-drawn versions of each digit). The test data consists of 10 images (1 hand-drawn version of each digit). Every image is of size 100x100 pixels. 

If you'd like to classify different types of images, you will have to train the parameters on a different set of training data. You may need to modify the way the pixel data is imported based on image size and OpenCV matrix type. The fundamentals of the cost function, gradient descent, and the sigmoid function will be the same, you'll just pass them different inputs.



## Explanation:

#### Data

**X** is a matrix of size (1000 x 10000). Each row holds all pixels for one training example. We have 1000 training images so there are 1000 rows. There is one column for each pixel. The images have been "flattened out" so that a (100x100) px image can be stored in a  (1x10000) row. The elements of the matrix depend on how you import the images using OpenCV. In this program, the images were imported as grayscale so there is only one channel (more on that shortly). In our case, each element of X has some value in the range 0 to 255 representing intensity (from black, through gray shades, to white). It just so happens that our training images only include white and black pixels to begin with (values 0 **or** 255), no greys. 

For work with coloured images, you may import using 3 channels (red, blue and green channels) so each entry in X would be a collection of 3 separate values in range 0 to 255. If that's the case, this program would have to be modified with regards to computations involving matrix X.

**y** is a matrix of size (1000 x 10). Each row is associated with (but does not contain) one image from X, in the same order that they appear in X. Each column is associated with (but does not contain in this context) one digit 0-9 figuratively speaking. For example, the element in y at row=566, col=8 is associated with the 566th image in X, and figuratively the digit 8. The elements in y are of value 0 or 1. In the example just laid out, 1 means that the image in X at row 566 **does represent** the digit 8, while 0 means that image **does not represent** the digit 8. Since our training images are arranged in X in order (100 zeroes, then 100 ones, etc) we know that row 566 of X should be an image of the digit 5, so y[566, 8]== 0. The contents of y are called labels. We do training by passing the program images with known values. y is how we know the values.

**theta** is a matrix of size (10 x 10000). Each row is associated with a digit 0-9. Each column is associated with one pixel. The elements of theta are the parameters that we train. Basically we'll have a guessing function where the parameters theta are the only variable inputs. We will iteratively modify those parameters until our function correctly guesses the training examples. Then when theta is used to guess an unseen training example, it should guess correctly. 



#### Components

**Sigmoid Function**
$$
g(z) = \frac{1}{1 + e^{-z}}
$$


Where *e* is Euler's number. A graph of the sigmoid function looks something like this:

![](/home/sam/downloads/graph.png)



We will want to know, is image *a* more likely a match for digit *b* (1) or a non-match (0), so the sigmoid function is a useful tool for producing results in the range 0 to 1.



**Hypothesis Function**
$$
h(x, \theta) = \frac{1}{1 + e^{-x \theta^{T}}}
$$
To return a prediction in the range 0 to 1, use this hypothesis function where:

- *x* is a row of *X* (all pixel values for one image)

- *θ* is a row of matrix theta (parameters associated with a figurative digit 0-9, one parameter for each pixel in *x*)

- *T* is the transpose operator used here on matrix *θ*

- The dot product x * θ_transpose is equivalent to:

  - $$
    \sum_{i=0}^{length(x)-1} x_iθ_i
    $$

This returns the probability, from 0 to 1, that image *x* is of the figurative digit represented by that row of *theta*. The returned probability will be accurate to the degree that the parameters *theta* have been trained correctly.

For example:

- Let *x* be the first row of *X* which is all pixel values for one drawing of the digit 0.
- Let *θ* be the first row of *theta* which is all parameters associated with the figurative digit 0.
- Note that the length of *x* and *θ*  are equal (both 10,000 in this program).
- We multiply each pixel value by its associated parameter, take the sum of those multiplications, and run them through the sigmoid function.
- Since the drawing is of the digit 0, and the *theta* row being used is associated with the figurative digit 0, we expect the output of the hypothesis to be close to 1, that is, if our parameters *theta* have been trained correctly. If we use a different row of *theta*, let's say associated with the figurative digit 7, we would expect the output of the hypothesis function to be less than in the former calculation because *x* is still an image of the digit 0.



**Cost Function**

How do we specify the degree to which our predictions are right or wrong?

- If y == 1

  - $$
    J(h(x,θ), y) = -ln(h(x, θ))
    $$

  - If the hypothesis (prediction) == 1 then the cost == 0. The prediction is in accordance with the label, so the margin of error is 0.
  - As the hypothesis (prediction) approaches 0, the cost approaches ∞.

- 

- If y == 0

  - $$
    J(h(x,θ), y) = -ln(1 - h(x,θ))
    $$

  - If the hypothesis (prediction) == 0 then the cost == 0. The prediction is in accordance with the label, so the margin of error is 0.
  - As the hypothesis (prediction) approaches 1, the cost approaches -∞.

The cost function uses a set of parameters to make a guess. It outputs 0 if the guess was correct, and deviates from 0 proportional to the wrongness of the prediction. 

For ease of use, we can rewrite the cost function on one line as follows:

Where *m* = number of training examples,
$$
J(h(x,θ), y) = -y*ln(h(x,θ)) - (1-y) * ln(1 - h(x, θ))
$$
If y == 0, LHS goes to 0

If y == 1, RHS goes to 0

So we get an equivalent equation on one line.

In our implementation, we pass the cost function one row of *theta*, one column of *y*, and the entire *X* matrix. The **full cost function** can be written like so:
$$
J(θ) = \frac{1}{m} 	\sum_{i=0}^{m-1} [-y_i*ln(h(x_i,θ)) - (1-y_i)*ln(1 - h(x_i,θ))]
$$

**Gradient Descent**

If the cost function determines that our prediction was wrong, how do we tune the parameters *theta* to get a better prediction?

We must minimize the cost *J(θ)*. Gradient descent is one way to do this.

![](/home/sam/downloads/gradient_graph.png)

The partial derivative of the cost *J(θ)* with respect to some *θ\_j* describes the slope at a point on the cost curve depicted above. If *θ\_j* is too small the slope will be negative, if *θ\_j* is too big the slope will be positive. If we subtract this slope from *θ\_j*, we will move in the right direction. Subtracting a negative adds to *θ\_j,* subtracting a positive makes *θ\_j* smaller, both of which cause us to step towards the minimum.
$$
θ_j = θ_j - \frac{\partial}{\partial θ_j}J(θ)
$$
We're stepping in the right direction, but how do we dictate the magnitude of the step? As each step approaches the minimum, the partial derivative value becomes smaller, because the slope is less steep. We will naturally subtract smaller and smaller values from *θ\_j* with each iteration. However, this may not produce small enough steps. If we step too far, we may overshoot the minimum. You may find your cost bouncing between values on either side of the minimum, but never converging at 0 cost. It may even start walking back *up*. We multiply this partial derivative by a **learning rate** *alpha* which dictates the size of the steps. If you find that you're bouncing between positive and negative costs, your learning rate is probably too high, set it lower. If you find that your gradient_descent program is taking too long to converge, your learning rate may be too low, try setting it higher to take bigger steps.


$$
θ_j = θ_j - \alpha \frac{\partial}{\partial θ_j}J(θ)
$$
For gradient descent, repeat the above algorithm, *simultaneously* updating all  *θ\_j* on each iteration. We want to make sure we use the old value of *θ\_j* to calculate *J(θ)* when we find the new *θ\_(j+1)*. Update *θ\_j* to *θ\_n*, store them somewhere, then simultaneously update that entire *theta* row before before proceeding to the next gradient descent iteration.

That partial derivative works out this way:
$$
\frac{\partial}{\partial θ_j}J(θ) = \frac{1}{m}\sum_{i=0}^{m-1}[(h(x_i,θ) - y_i)x_{ji}]
$$


**Overfitting and the Regularization Solution**

![](/home/sam/downloads/overfitting.png)

Overfitting is when your model (in our case logistic regression) fits the training data too well. That being the case, it may fail to generalize with regards to new data it's designed to compute. Underfitting is when the model doesn't fit the training data well enough, and so it's not a good tool for making predictions on new inputs. We want our model to fit the data "just right".

Overfitting tends to happen when we have too many features. Generally, each feature is represented by one column of *X*. In this program, each pixel is a feature, and we have 10,000 of them. Meanwhile we only have 100 training examples for each digit. We are at risk of overfitting the data and failing to generalize when making predictions on new digits.

One solution may be to reduce the number of features. We may like to gather (5x5) groups of pixels and average them out in some way. In this case it works well to do: if any pixel in the (5x5) group is black, convert the entire (5x5) group to one black pixel (because there is so much white this works best). That leaves us with a (20x20) pixel images. Flattened out, that's only 400 features. A big reduction from 10,000! This type of solution may not work if, for example, some images become too blurry and the result is an indistinguishable mess.

A second solution is called regularization. We can reduce the magnitude of our theta values to soften the impact they have on our model. By convention we use *λ* to represent our regularization term. When we add a large regularization value to the cost, gradient descent must find smaller values of theta in order to converge on the minimum. That's how, maybe unintuitively, adding gets us smaller parameters. Note that we don't regularize *θ\_0*.

Where *j* == number of features, at the end of the cost function we add:
$$
+\frac{λ}{2m}\sum_{j=0}^{j-1}\theta_j^2
$$
This causes the partial derivative of the cost function to change. We have to update gradient descent because we now get that:
$$
\frac{\partial}{\partial θ_j}J(θ) = \frac{1}{m}\sum_{i=0}^{m-1}[(h(x_i,θ) - y_i)x_{ji}] + \frac{λ}{m}\theta_j
$$
**Multiple Class Classification**

For classifying multiple classes (in this case digits 0-9) we just build multiple classifiers. This is why we have 10 columns of *y* and 10 rows of *theta*.