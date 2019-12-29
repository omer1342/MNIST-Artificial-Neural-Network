'''
AUTHOR: Omer Hen
PURPOSE: Final Computer Science Project, 2019, Shimon Ben Zvi High School

This project covers the implementation and research into ANNs.
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip, urllib.request
import os.path
import cProfile, pstats


def profile(fnc):
    '''
    Profile for single function.
    This function is based on Python's documention for profiling:
    https://docs.python.org/2/library/profile.html

    Arguments:
    fnc -- function to be tested

    Returns:
    inner -- the output of the function: "fnc", in order for the main program to continue running
    '''

    def inner(*args, **kwargs):
        '''
        Inner workings of the profiler function.

        Arguments:
        *args, **kwargs -- arguments on the function: "fnc"

        Returns:
        retval -- the output of the functon: "fnc" in order for the main program to continue running
        '''
        pr = cProfile.Profile() # Creating a new cProfile object
        pr.enable() # Start recording
        retval = fnc(*args, **kwargs)
        pr.disable() #Stop recording
        ps = pstats.Stats(pr).strip_dirs().sort_stats('cumtime') # Retrieve the info, hide dirs and sory by cumulative time
        ps.print_stats()                                         # Print stats to console
        return retval

    return inner


def layer_sizes(X, Y):
    """
    Calculates the required number of neurons in the input and output layers.

    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0] 
    n_y = Y.shape[0]
    
    return n_x, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    Creates the initial weights and biases for the layers of the network.

    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing the network's parameters:
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
    Mimics the Feedforward Propagation of a neural network. 

    Argument:
    X -- input data of size (784, number of examples)
    parameters -- python dictionary containing the network's parameters: W1, b1, W2 and b2 (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve the weights and biases from "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1     # Calculation of the weighted sum plus the bias
    A1 = ReLU(Z1)            # tanh activation function in the hidden layer
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)            # sigmoid activation function in the output layer 

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y):
    """
    Computes the cost, according to the cross-entropy equation.

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (10, number of examples) 
    Y -- "true" labels vector, or simply the target values of shape (10, number of examples)                          
    parameters -- python dictionary containing the network's parameters: W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1] # Number of examples

    # Compute cost according to the cross-entropy equation
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1- A2))
    cost = -np.sum(logprobs)/m
    
    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implements Backward Propagation for the entire network.

    Arguments:
    parameters -- python dictionary containing the network's parameters: W1, b1, W2 and b2 
    cache -- a dictionary containing: "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (784, number of examples)
    Y -- "true" labels vector, or simply the target values of shape (10, number of examples)  

    Returns:
    nabla -- python dictionary containing the gradients with respect to different parameters
    """
    m = X.shape[1] # Number of examples

    # Retrieve "W1" and "W2" from the dictionary "parameters"
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve "Z1", "A1" and "A2" from the dictionary "cache"
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims = True) / m

    dZ1 = np.multiply(np.dot(W2.T, dZ2), ReLU_der(Z1))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    nabla = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return nabla


def update_parameters(parameters, nabla, learning_rate):
    """
    Updates parameters using the Gradient Descent update rule.

    Arguments:
    parameters -- python dictionary containing the network's parameters: W1, b1, W2 and b2 
    nabla -- python dictionary containing the network parameters' gradients
    learning_rate -- the learning rate of the network, affects the impact the gradients will have on updated parameters

    Returns:
    parameters -- python dictionary containing the updated parameters 
    """
    # Retrieve the weights and biases from "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve the gradients from "nabla"
    dW1 = nabla["dW1"]
    db1 = nabla["db1"]
    dW2 = nabla["dW2"]
    db2 = nabla["db2"]

    # Update the network's parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(data, n_h, learning_rate = 0.35, epochs = 10000, print_cost=True):
    """
    Main function, brings together the network.
    
    Arguments:
    data -- python dictionary containing the training and tetsing data:
            X -- dataset of shape (784, number of examples)
            Y -- labels of shape (10, number of examples)
            test_X -- dataset of shape (784, number of examples)
            test_Y -- labels of shape (10, number of examples)
    n_h -- size of the hidden layer
    learning_rate -- the learning rate of the network, affects the impact the gradients will have on updated parameters
    num_iterations -- Number of epochs
    print_cost -- if True, print the cost every 500 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict 
    """
    # Retrieve training and testing data from the dictionary "data"
    X = data["X"]
    Y = data["Y"]
    test_X = data["test_X"]
    test_Y = data["test_Y"]

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]

    # Initialize parameters, then retrieve W1, b1, W2, b2
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    loss = []
    accuracy = []

    # Loop (Gradient Descent)
    for i in range(0, epochs):
        
        # First, forward propagate through the network 
        A2, cache = forward_propagation(X, parameters)

        # Then, compute the cost of the network for its final output: "A2" and target values: "Y"
        cost = compute_cost(A2, Y)

        # Backpropagate through the layers of the network in order to achieve the gradients 
        nabla = backward_propagation(parameters, cache, X, Y)

        # Update the weights and biases according to the gradients
        parameters = update_parameters(parameters, nabla, learning_rate)

        if print_cost and i % 500 == 0:
            precision = evaluate(test_X, test_Y, parameters)
            print('Cost after iteration %i: %f, %a percent accuracy.' %(i, cost, precision))
            loss.append(cost)
            accuracy.append(precision)

    visualize(loss, accuracy, epochs)
    return parameters


def evaluate(X, y, parameters):
    '''
    Evaluates the network for its level of accuracy in testing data.

    Arguments:
    X -- dataset of shape (784, number of examples)
    y -- labels of shape (10, number of examples)
    parametes -- python dictionary containing the network's parameters: W1, b1, W2 and b2

    Returns:
    A rounded float describing the percentage of testing examples the network classified correctly
    '''
    # First, get the network output of the testing data
    o, cache = forward_propagation(X, parameters)
    # Then, compare the network output to the target values 
    return round(((y.T.argmax(axis = 1) == o.T.argmax(axis = 1)).sum() / y.shape[1] * 100), 3)
    

def load_data():
    '''
    Loads the training and testing data from the MNIST databse.

    Arguments:
    None.

    Returns:
    data -- python dictionary containing the training/testing examples and target values
    '''
    # File name of desired database
    file_name = 'mnist.pkl.gz'

    # If database file is not already downloaded, download it
    if (not os.path.isfile(file_name)):
        urllib.request.urlretrieve("http://deeplearning.net/data/mnist/" + file_name, file_name)

    # Extract the training, testing and validating examples from the database
    with gzip.open(file_name, 'rb') as file:
        trs, vs, ts = pickle.load(file, encoding='latin1')

    # Format the training and testing data in a proper manner - in order for the network to be able to interact with it
    training_images = np.array(trs[0])
    training_labels = np.array([vectorize(y) for y in trs[1]])

    training_images, training_labels = randomize(training_images, training_labels) # Shuffle the data

    test_images = np.array(ts[0])
    test_labels = np.array(np.array([vectorize(y) for y in ts[1]]))

    test_images, test_labels = randomize(test_images, test_labels)
    
    data = {"X" : training_images,
            "Y" : training_labels,
            "test_X" : test_images,
            "test_Y" : test_labels}
    
    return data
  
def vectorize(j):
    '''
    Vectorizes an integer according to the output layer's requirements.

    Arguments:
    j -- An integer describing the index of the right answer of a certain target value

    Returns:
    a -- A vector describing the target value, based on j, for a certain example
    '''
    # Generates an empty array - all wrong answers are denoted wiht 0.0, and marks the right answer with 1.0
    a = np.zeros(10)
    a[j] = 1.0
    return a


def sigmoid(x):
    '''
    The sigmoid function.

    Arguments:
    x -- A certain value, or matrix to undergo the sigmoid function

    Returns
    The output of the sigmoid function for 'x'
    '''
    return 1 / (1 + np.exp(-x))


def randomize(x, y):
    '''
    Shuffles the arrays "x" and "y" in a random manner while keeping the
    corresponding indices of the elements in arrays.

    Arguments:
    x -- Numpy Array of n dimensions
    y -- Numpy array of n dimensions

    Returns:
    Transposed and shuffled "x" and "y"
    '''
    gamma = np.arange(x.shape[0])
    np.random.shuffle(gamma)
    return x[gamma].T, y[gamma].T


def visualize(loss, evaluations, epochs):
    '''
    Plots the data collected during the training process of the network into graphs
    displaying the cost and accuracy level of the network throughout the process.

    Arguments:
    loss -- Array containing the cost of the network over the course of the training process
    evluations -- Array containing the accuracy level of the network over the course of the training process
    epochs -- Number of epochs the network underwent

    Returns:
    None, produces graphs and saves them to the local folder
    '''
    xaxis = np.arange(0, epochs, 500)

    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(xaxis, loss)

    plt.savefig('CostGraph.png') # Save cost graph to local folder

    plt.cla() # Clear figure for the next graph to be plotted
    plt.grid()
    plt.hlines(100, 0, epochs, linestyles='dashed') # Plot maximum line - at 100% success rate
    plt.ylabel('Accuracy')
    plt.plot(xaxis, evaluations)

    plt.savefig('EvalGraph.png') # Save evluation graph to local folder

def ReLU(x):
    '''
    The ReLU activation function.

    Arguments:
    x -- Number or array to perform the ReLU function on

    Returns:
    The output of the ReLU function for: "x"
    '''
    return np.maximum(0,x)


def ReLU_der(x):
    '''
    The derivative of the ReLU functon.

    Arguments:
    x -- Number or array to perform the derivative of the ReLU function on

    Returns:
    The output of the derivative of the ReLU function
    '''
    return np.where(x <=0, 0, 1)


if __name__ == "__main__":
    # Retrieve the training and testing data
    data = load_data()
    # Calls the main method to start the learning proces
    nn_model(data, 150)



