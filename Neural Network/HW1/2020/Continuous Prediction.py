# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 22:34:08 2020

@author: hossein-pc
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow.python.framework import ops
import math



# =============================================================================
# Creates a list of random minibatches from (X, Y)
# =============================================================================

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# =============================================================================
# load data
# =============================================================================

random.seed(1)

DATADIR = "C:/Users/Hossein/Downloads/Uni/Shiraz Uni/Neural Network/HW/HW1/NNDL_Pr1"
CATEGORIES = ["Train Data","Test Data"]

IMG_SIZE = 100
training_data = []
testing_data = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            if category =="Train Data":
                training_data.append(new_array)
            else:
                testing_data.append(new_array)
        except Exception as e:
            pass

# =============================================================================
# prepricessing & shuffling data set
# =============================================================================
            
training_label = [1]*10+[2]*10+[3]*10+[4]*10+[5]*10+[0]*10
testing_label = [4]*5+[5]*5+[3]*5+[1]*5+[0]*5+[2]*5

traning_data_set = []
testing_data_set = []

for i in range(60):
    traning_data_set.append([training_data[i],training_label[i]])
    
for i in range(30):
    testing_data_set.append([testing_data[i],testing_label[i]])

random.shuffle(traning_data_set)
random.shuffle(testing_data_set)

X_train = []
Y_train = []
X_test = []
Y_test = []

for i in range(60):
    X_train.append(traning_data_set[i][0])
    Y_train.append(traning_data_set[i][1])
    
for i in range(30):
    X_test.append(testing_data_set[i][0])
    Y_test.append(testing_data_set[i][1])
    
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train).reshape(60,1)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test).reshape(30,1)

# =============================================================================
# Example of a picture
# =============================================================================

#index = 58
#plt.imshow(X_train[index])
#plt.show()
#print ("y = " + str(np.squeeze(Y_train[index])))

# =============================================================================
# Flatten the training and test images
# =============================================================================

X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T

# =============================================================================
# Normalize image vectors
# =============================================================================

X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.

# =============================================================================
# Convert training and test labels to one hot matrices
# =============================================================================

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

Y_train = convert_to_one_hot(Y_train, 6)
Y_test = convert_to_one_hot(Y_test, 6)

#print("number of training examples = " + str(X_train.shape[1]))
#print("number of test examples = " + str(X_test.shape[1]))
#print("X_train shape: " + str(X_train.shape))
#print("Y_train shape: " + str(Y_train.shape))
#print("X_test shape: " + str(X_test.shape))
#print("Y_test shape: " + str(Y_test.shape))

# =============================================================================
# create_placeholders
# =============================================================================

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    
    
    return X, Y

#X, Y = create_placeholders(10000, 6)
#print("X = " + str(X))
#print("Y = " + str(Y))

# =============================================================================
# initialize_parameters
# =============================================================================
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [50, 10000]
                        b1 : [50, 1]
                        
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, w2, b2
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    
    W1 = tf.get_variable("W1", [50, 10000], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [50, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [6, 50], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [6, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

#tf.reset_default_graph()
#with tf.Session() as sess:
#    parameters = initialize_parameters()
#    print("W1 = " + str(parameters["W1"]))
#    print("b1 = " + str(parameters["b1"]))
    
# =============================================================================
# forward_propagation
# =============================================================================

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2"
                  the shapes are given in initialize_parameters

    Returns:
    Z2 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    
                  
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    
    return Z2

tf.reset_default_graph()

#with tf.Session() as sess:
#    X, Y = create_placeholders(10000, 6)
#    parameters = initialize_parameters()
#    Z2 = forward_propagation(X, parameters)
#    print("Z2 = " + str(Z2))

# =============================================================================
# compute_cost  
# =============================================================================

def compute_cost(Z2, Y):
    """
    Computes the cost
    
    Arguments:
    Z2 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z2
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z2)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    
    return cost

#tf.reset_default_graph()
#with tf.Session() as sess:
#    X, Y = create_placeholders(10000, 6)
#    parameters = initialize_parameters()
#    Z2 = forward_propagation(X, parameters)
#    cost = compute_cost(Z2, Y)
#    print("cost = " + str(cost))
    
# =============================================================================
# building the model
# =============================================================================

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 500, minibatch_size = 1, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 10000, number of training examples = 60)
    Y_train -- test set, of shape (output size = 6, number of training examples = 60)
    X_test -- training set, of shape (input size = 10000, number of training examples = 30)
    Y_test -- test set, of shape (output size = 6, number of test examples = 30)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z2 = forward_propagation(X, parameters)
    
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z2, Y)
    
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train})*100, "%")
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test})*100, "%")
        
        return parameters

parameters = model(X_train, Y_train, X_test, Y_test)

# =============================================================================
# predict
# =============================================================================

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2"
                  the shapes are given in initialize_parameters
    Returns:
    Z2 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
     
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    
    return Z2


def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
    
    x = tf.placeholder("float", [10000, 1])
    
    z2 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z2)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction


path1 = os.path.join(DATADIR, "some pics")
img1 = os.listdir(path1)

img_index = 21
my_image_prediction = predict(X_test[:,img_index].reshape(10000,1), parameters)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
plt.imshow(X_test[:,img_index].reshape(100,100))


