import numpy as np
import copy


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of z
    :param z: A scalar or numpy array of any size.
    :return: sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim: int) -> tuple[np.ndarray, float]:
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    :param dim: size of the w vector we want (or number of parameters in this case)
    :return:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    """
    w = np.zeros((dim, 1))
    b = .0

    return w, b


def propagate(w: np.ndarray, b: float, X: np.ndarray, Y: np.ndarray) -> tuple[dict, float]:
    """
    Implement the cost function and its gradient for the propagation explained above
    :param w: weights, a numpy array of size (height_px * width_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of size (height_px * width_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    :return:
    grads -- dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)

    # Compute cost function
    cost = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w: np.ndarray, b: float, X: np.ndarray, Y: np.ndarray, num_iterations: int = 100,
             learning_rate: float = 0.009, print_cost: bool = False) -> tuple[dict, dict, float]:
    """
    This function optimizes w and b by running a gradient descent algorithm
    :param w: weights, a numpy array of size (height_px * width_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of shape (height_px * width_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    :param num_iterations: number of iterations of the optimization loop
    :param learning_rate: learning rate of the gradient descent update rule
    :param print_cost: True to print the loss every 100 steps
    :return:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # Update weights
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    :param w: weights, a numpy array of size (height_px * width_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of size (height_px * width_px * 3, number of examples)
    :return: Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a figure or table being present in the page
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    return Y_prediction


def model(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, num_iterations: int = 2000,
          learning_rate: float = 0.5, print_cost: bool = False) -> dict:
    """
    Builds the logistic regression model by calling the function we've implemented previously
    :param X_train: training set represented by a numpy array of shape (height_px * width_px * 3, m_train)
    :param Y_train: training labels represented by a numpy array (vector) of shape (1, m_train)
    :param X_test: test set represented by a numpy array of shape (height_px * width_px * 3, m_test)
    :param Y_test: test labels represented by a numpy array (vector) of shape (1, m_test)
    :param num_iterations: hyperparameter representing the number of iterations to optimize the parameters
    :param learning_rate: hyperparameter representing the learning rate used in the update rule of optimize()
    :param print_cost: Set to True to print the cost every 100 iterations
    :return: d -- dictionary containing information about the model.
    """

    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations=num_iterations, learning_rate=learning_rate,
                                    print_cost=print_cost)

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d
