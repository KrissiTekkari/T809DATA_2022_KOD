# The Back-propagation Algorithm
The aim of this exercise is to implement the back-propagation algorithm. There are many ready-made tools that do this already but here we aim to implement the algorithm using only the linear algebra and other mathematics tools available in numpy and scipy. This way you get an insight into how the algorithm works before you move on to use it in larger toolboxes.

We will restrict ourselves to fully-connected feed forward neural networks with one hidden layer (plus an input and an output layer).

## Section 1

### Section 1.1 The Sigmoid function
We will use the following nonlinear activation function:

$$\sigma(a)=\frac{1}{1+e^{-a}}$$

We will also need the derivative of this function:

$$\frac{d}{da}\sigma(a) = \frac{e^{-a}}{(1+e^{-a})^2} = \sigma(a) (1-\sigma(a))$$

Create these two functions:
1. The sigmoid function: `sigmoid(x)`
2. The derivative of the sigmoid function: `d_sigmoid(x)`

**Note**: To avoid overflows, make sure that inside `sigmoid(x)` you check if `x<-100` and return `0.0` in that case.

Example inputs and outputs:
* `sigmoid(0.5)` -> `0.6224593312018546`
* `d_sigmoid(0.2)` -> `0.24751657271185995`

### Section 1.2 The Perceptron Function
A perceptron takes in an array of inputs $X$ and an array of the corresponding weights $W$ and returns the weighted sum of $X$ and $W$, as well as the result from the activation function (i.e. the sigmoid) of this weighted sum. *(see Eq. 5.48 & 5.62 in Bishop)*.

Implement the function `perceptron(x, w)` that returns the weighted sum and the output of the activation function.

Example inputs and outputs:
* `perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1]))` -> `(1.0799999999999998, 0.7464939833376621)`
* `perceptron(np.array([0.2,0.4]),np.array([0.1,0.4]))` -> `(0.18000000000000005, 0.5448788923735801)`

### Section 1.3 Forward Propagation
When we have the sigmoid and the perceptron function, we can start to implement the neural network.

Implement a function `ffnn` which computes the output and hidden layer variables for a single hidden layer feed-forward neural network. If the number of inputs is $D$, the number of hidden layer neurons is $M$ and the number of output neurons is $K$, the matrices $W_1$ of size $[(D+1)\times M]$ and $W_2$ of size $[(M+1)\times K]$ represent the linear transform from the input layer and the hidden layer and from the hidden layer to the output layer respectively.

Write a function `y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)` where:

* `x` is the input pattern of $(1\times D)$ dimensions (a line vector)
* `W1` is a $((D+1)\times M)$ matrix and `W2` is a $(M+1)\times K$ matrix. (the `+1` are for the bias weights)
* `a1` is the input vector of the hidden layer of size $(1\times M)$ (needed for backprop).
* `a2` is the input vector of the output layer of size $(1\times K)$ (needed for backprop).
* `z0` is the input pattern of size $(1\times (D+1))$, (this is just `x` with `1.0` inserted at the beginning to match the bias weight).
* `z1` is the output vector of the hidden layer of size $(1\times (M+1))$ (needed for backprop).
* `y` is the output of the neural network of size $(1\times K)$.

Example inputs and outputs:

*First load the iris data:*
```
features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)
```

*Then call the function*
```
# initialize the random generator to get repeatable results
np.random.seed(1234)

# Take one point:
x = train_features[0, :]
K = 3 # number of classes
M = 10
D = 4
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
```
*Outputs*:
* `y` : `[0.08510131, 0.25863437, 0.92845325]`
* `z0`: `[1. , 6.3, 2.5, 4.9, 1.5]`
* `z1`: `[1.   , 0.31277934, 0.32324425, 0.84118978, 0.40816423, 0.96696017, 0.95999698, 0.0063807 , 0.56681781, 0.9137305 , 0.9999515 ]`
* `a1`: `[-0.78715749, -0.73890218,  1.6671074 , -0.37155955,  3.37644349,3.17797518, -5.04807589,  0.26887954,  2.36005949,  9.93383443]`
* `a2`: `[-2.37497091, -1.05307858,  2.56316901]`

### Section 1.4 Backward Propagation

We will now implement the back-propagation algorithm to evaluate the gradient of the error function $\Delta E_n(x)$.

Create a function `y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)` where:
* `x, M, K, W1` and `W2` are the same as for the `ffnn` function
* `target_y` is the target vector. In our case (i.e. for the classification of Iris) this will be a vector with 3 elements, with one element equal to 1.0 and the others equal to 0.0. (*).
* `y` is the output of the output layer (vector with 3 elements)
* `dE1` and `dE2` are the gradient error matrices that contain $\frac{\partial E_n}{\partial w_{ji}}$ for the first and second layers.

Assume sigmoid hidden and output activation functions and assume cross-entropy error function (for classification). Notice that $E_n(\mathbf{w})$ is defined as the error function for a single pattern $\mathbf{x}_n$. *The algorithm is described on page 244 in Bishop*.

The inner working of your `backprop` function should follow this order of actions:
1. run `ffnn` on the input.
2. calculate $\delta_k = y_k - target\_y_k$
3. calculate $\delta_j = (\frac{d}{da}\sigma(a^{(1)}_j)) \sum_{k} w_{k,j+1}\delta_k$ (the `+1` is because of the bias weights)
4. initialize `dE1` and `dE1` as zero-matrices with the same shape as `W1` and `W2`
5. calculate `dE1_{i,j}` $= \delta_j z^{(0)}_i$  and `dE2_{j,k}` = $\delta_k z^{(1)}_j$

Example inputs and outputs:

*Call the function*
```
# initialize random generator to get predictable results
np.random.seed(42)

K = 3  # number of classes
M = 6
D = train_features.shape[1]

x = features[0, :]

# create one-hot target for the feature
target_y = np.zeros(K)
target_y[targets[0]] = 1.0

# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1

y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
```
*Output*
* `y`: `[0.42629045 0.1218163  0.56840796]`
* `dE1`:
```
[[-3.17372897e-03  3.13040504e-02 -6.72419861e-03  7.39219402e-02
  -1.16539047e-04  9.29566482e-03]
 [-1.61860177e-02  1.59650657e-01 -3.42934129e-02  3.77001895e-01
  -5.94349138e-04  4.74078906e-02]
 [-1.11080514e-02  1.09564176e-01 -2.35346951e-02  2.58726791e-01
  -4.07886663e-04  3.25348269e-02]
 [-4.44322055e-03  4.38256706e-02 -9.41387805e-03  1.03490716e-01
  -1.63154665e-04  1.30139307e-02]
 [-6.34745793e-04  6.26081008e-03 -1.34483972e-03  1.47843880e-02
  -2.33078093e-05  1.85913296e-03]]
```
* `dE2`:
```
[[-5.73709549e-01  1.21816299e-01  5.68407958e-01]
 [-3.82317044e-02  8.11777445e-03  3.78784091e-02]
 [-5.13977514e-01  1.09133338e-01  5.09227901e-01]
 [-2.11392026e-01  4.48850716e-02  2.09438574e-01]
 [-1.65803375e-01  3.52051896e-02  1.64271203e-01]
 [-3.19254175e-04  6.77875452e-05  3.16303980e-04]
 [-5.60171752e-01  1.18941805e-01  5.54995262e-01]]
```

> (*): *This is referred to as [one-hot encoding](https://en.wikipedia.org/wiki/One-hot). If we have e.g. 3 possible classes: `yellow`, `green` and `blue`, we could assign the following label mapping: `yellow: 0`, `green: 1` and `blue: 2`. Then we could encode these labels as:*
$$
\text{yellow} = \begin{bmatrix}1\\0\\0\end{bmatrix}, \text{green} = \begin{bmatrix}0\\1\\0\end{bmatrix}, \text{blue} = \begin{bmatrix}0\\0\\1\end{bmatrix},
$$
>*But why would we choose to do this instead of just using $0, 1, 2$ ? The reason is simple, using ordinal categorical label injects assumptions into the network that we want to avoid. The network might assume that `yellow: 0` is more different from `blue: 2` than `green: 1` because the difference in the labels is greater. We want our neural networks to output* **probability distributions over classes** *meaning that the output of the network might look something like:*
$$
\text{NN}(x) = \begin{bmatrix}0.32\\0.03\\0.65\end{bmatrix}
$$
> *From this we can directly make a prediction, `0.65` is highest so the model is most confident in that the input feature corresponds to the `blue` label*


## Section 2 - Training the Network
We are now ready to train the network. Training consists of:
1. forward propagating an input feature through the network
2. Calculate the error between the prediction the network made and the actual target
3. Back-propagating the error through the network to adjust the weights.


### Section 2.1
Write a function called `W1tr, W2tr, E_total, misclassification_rate, guesses = train_nn(X_train, t_train, M, K, W1, W2, iterations, eta)` where

Inputs:
* `X_train` and `t_train` are the training data and the target values
* `M, K, W1, W2` are defined as above
* `iterations` is the number of iterations the training should take, i.e. how often we should update the weights
* `eta` is the learning rate.

Outputs:
* `W1tr`, `W2tr` are the updated weight matrices
* `E_total` is an array that contains the error after each iteration.
* `misclassification_rate` is an array that contains the misclassification rate after each iteration
* `guesses` is the result from the last iteration, i.e. what the network is guessing for the input dataset `X_train`.

The inner working of your `train_nn` function should follow this order of actions:

1. Initialize necessary variables
2. Run a loop for `iterations` iterations.
3. In each iteration we will collect the gradient error matrices for each data point. Start by initializing `dE1_total` and `dE2_total` as zero matrices with the same shape as `W1` and `W2` respectively.
4. Run a loop over all the data points in `X_train`. In each iteration we call backprop to get the gradient error matrices and the output values.
5. Once we have collected the error gradient matrices for all the data points, we adjust the weights in `W1` and `W2`, using `W1 = W1 - eta * dE1_total / N` where `N` is the number of data points in `X_train` (and similarly for `W2`).
6. For the error estimation we'll use the cross-entropy error function, *(Eq. 4.90 in Bishop)*.
7. When the outer loop finishes, we return from the function

Example inputs and outputs:

*Call the function*:
```
# initialize the random seed to get predictable results
np.random.seed(1234)

K = 3  # number of classes
M = 6
D = train_features.shape[1]

# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
    train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
```
*Output*
* `W1tr`:
```
[[-0.98950665,  0.2377717 , -0.00477265,  0.54839507,  0.75946904, -0.32423704],
[-1.13990844,  0.57545996,  0.82422467,  0.65281234,  0.29731528, 0.36236617],
[-0.49057592,  0.40399158,  0.38152979,  0.04819451,  0.8895356 , -0.81313404],
[ 1.42962024,  0.76166834, -1.76098208,  0.21569728, -1.75766404, -0.18698651],
[ 1.74935443,  0.30337755, -0.85651739,  0.57883159, -0.76902497, -0.05523648]]
```
* `W2tr`:
```
[[ 0.54871673, -0.47911188, -0.00984517],
 [-3.07944628, -0.50625841,  2.46043706],
 [-0.76538278,  0.50367072, -0.72183744],
 [ 2.23493305, -1.49004112, -1.95795014],
 [-1.12456261,  0.00424719, -0.40165383],
 [ 2.19121834, -1.68374312, -1.16334662],
 [-0.39521195,  0.18956774, -0.93271377]]
```
* `Etotal`:
```
[2.08789826, 2.01463545, 1.95876231, 1.91540916, 1.88107942
...
0.82226302, 0.82154169, 0.82081961, 0.82009677, 0.81937318]
```
* `misclassification_rate`:
```
[0.7 , 0.7 , 0.7 , 0.7 , 0.65, 0.1 , 0.35, 0.3 , 0.3 , 0.3 , 0.3 ,
...
0.1 , 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 ]
```
* `last_guesses`:
```
[2., 2., 2., 1., 0., 2., 1., 0., 0., 1., 2., 0., 1., 2., 2., 2., 0., 0., 1., 0.]
```

### Section 2.2
Write a function `guesses = test_nn(X_test, M, K, W1, W2)` where

* `X_test`: the dataset that we want to test.
* `M`: size of the hidden layer.
* `K`: size of the output layer.
* `W1, W2`: fully trained weight matrices, i.e. the results from using the `train_nn` function.
* `guesses`: the classification for all the data points in the test set `Xtest`. This should be a $(1\times N)$ vector where $N$ is the number of data points in `X_test`.

The function should run through all the data points in `X_test` and use the `ffnn` function to guess the classification of each point.

### Section 2.3
**You should answer this question in a PDF document**
Now train your network and test it on the Iris dataset. Use 80% (as usual) for training.
1. Calculate the accuracy
2. Produce a confusion matrix for your test features and test predictions
3. Plot the `E_total` as a function of iterations from your `train_nn` function.
4. Plot the `misclassification_rate` as a function of iterations from your `train_nn` function.

Present this and relevant discussion in a single PDF document.

### Independent section
In this assignment we have created a naive neural network. This network could be changed in small ways that might have meaningful impact on performance. To name a few:
1. The size of the hidden layer
2. The learning rate
3. The initial values for the weights

Change the network in some way and compare the accuracy of your network for different configurations (i.e. if you decide to change the size of the hidden layer, a graph of test accuracy as a function of layer size would be ideal).

Use the PDF document to present your solution to the independent section.