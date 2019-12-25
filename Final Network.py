import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip, urllib.request, json
import numpy as np
import os.path

np.random.seed(1) # set a seed so that the results are consistent



def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (number of examples, input size)
    Y -- labels of shape (number of examples, output size)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0] 
    n_y = Y.shape[0]
    
    return n_x, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cost(cross- entropy cost) given in the equation above.

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples) / (number of examples, 1)
    Y -- "true" labels vector of shape (1, number of examples)                          / (number of examples, 1)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost --cross-entropy cost
    """

    m = Y.shape[1] # number of examples

    # Compute cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1- A2))
    cost = -np.sum(logprobs)/m
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
                                # E.g., turns [[17]] into 17

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims = True) / m

    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.square(A1)))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate = 0.35):
    """
    Updates parameters using the gradient accent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, test_X, test_Y, num_iterations = 20000, print_cost=True):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)


        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print('Cost after iteration %i: %f, %a percent accuracy.' %(i, cost, round(evaluate(test_X, test_Y, parameters)*100, 3)))

    return parameters



def evaluate(X, y, parameters):

    o, cache = forward_propagation(X, parameters)
    return (y.T.argmax(axis = 1) == o.T.argmax(axis = 1)).sum() / y.shape[1]
    



def load_data():
  file_name = 'mnist.pkl.gz'

  if (not os.path.isfile(file_name)):
    urllib.request.urlretrieve("http://deeplearning.net/data/mnist/" + file_name, file_name)
    
  with gzip.open(file_name, 'rb') as file:
    trs, vs, ts = pickle.load(file, encoding='latin1')

  training_images = np.array(trs[0])
  training_labels = np.array([vectorize(y) for y in trs[1]])

  test_images = np.array(ts[0])
  test_labels = np.array(np.array([vectorize(y) for y in ts[1]]))


  return training_images, training_labels, test_images, test_labels
  
def vectorize(k):
  a = np.zeros(10)
  a[k] = 1.0
  return a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
  images, labels, ts_images, ts_labels = load_data()
  img = images[:45000].T
  lbl = labels[:45000].T

  tsimg = ts_images.T
  tlbl = ts_labels.T

  nn_model(img, lbl, 30, tsimg, tlbl)



