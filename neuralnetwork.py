

"""
## Part 1 - Neural Network using NumPy

In this part we learn to:
1. Load and preprocess datasets.
2. Implement and train a neural network (multi-layer perceptron) for handwriting recognition (MNIST dataset), using numpy only.

### **1. Dataset**

Import useful packages
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

"""Download the MNIST dataset"""

X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)

"""Data normalization"""

##  Normalize the dataset according to Min-Max normalization.
def min_max_norm(data):
    data_min = np.min(data)
    data_max = np.max(data)

    # Calculate the range of the data
    data_range = data_max - data_min

    # Initialize a list to store the normalized values
    normalized_data = []

    # Normalize each value in the data
    for value in data:
        normalized_value = (value - data_min) / data_range
        normalized_value = normalized_value
        normalized_data.append(normalized_value)
    return np.array(normalized_data)

X = min_max_norm(X)

"""Split the data into Train set and Test set"""


# Shuffle the data
permuted_indices = np.random.permutation(len(X)) #7000
shuffled_X = X[permuted_indices]
shuffled_y = y[permuted_indices]

# Split the data into a training set and a test set
split_index = int(len(shuffled_X) * 0.8)
X_train = shuffled_X[:split_index]
y_train = shuffled_y[:split_index]
X_test = shuffled_X[split_index:]
y_test = shuffled_y[split_index:]

"""Activation function"""

##  Implement the sigmoid activation function and its derivative
def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
  s = sigmoid(z)
  return s * (1 - s)

"""Softmax function"""

## Implement the softmax function

def softmax(z):
  #substracting max values in z - creating equilavent expression that treats numerical instability
  e_z = np.exp(z - np.max(z))
  return e_z / e_z.sum()

"""Loss function"""

## Negative Log Likelihood loss function for the multiclass

def nll_loss(y_pred, y):
  loss = -np.sum(y * np.log(y_pred))
  return loss / float(y_pred.shape[0])

"""Hyper-Parameters"""

## define the main hyper-parameters.
learning_rate = 0.01
num_of_epochs = 10

"""Parameters initialization"""

## initialize the parameters (MNIST dataset has 10 classes).
n_hidden = 128
n_outputs = 10
n_inputs = 784
w1 = np.random.randn(n_inputs, n_hidden)
b1 = np.zeros(n_hidden)
w2 = np.random.randn(n_hidden, n_outputs)
b2 = np.zeros(n_outputs)

### **2. Training**


def train(X, y, num_of_epochs):
  global w2, b2, w1, b1
  train_size = len(X)

  for epoch in range(num_of_epochs):
    avg_epoch_loss = 0
    for i in range(train_size):
      #  Forward propagation
      z1 = np.dot(X[i],w1) + b1
      h1 = sigmoid(z1)
      Z2 = np.dot(h1,w2)+b2
      y_hat = softmax(Z2)
      if i == 0 and epoch == 0:
        y_hat = y_hat.tolist()
        y_hat = np.array([y_hat])

      y_true = y[i]
      labels_one_hot = np.zeros(y_hat.shape)  #make y_true vector of 1 in the right class and 0 in the others
      labels_one_hot[0,int(y_true)] = 1
      y_true = labels_one_hot

      #Compute loss
      loss =  nll_loss(y_hat, y_true)
      avg_epoch_loss = avg_epoch_loss + loss


      # Back propagation - compute the gradients of each parameter
      dZ2 = (y_hat - y_true)  #shape of (1, 10)

      if i == 0 and epoch == 0:
        temp3 = h1.tolist()  #convert shape of h1 from (128,) to (1,128) - technical
        h1=np.array([temp3])

      dW2 = np.dot(h1.T, dZ2)  #shape (128,10)
      db2 = dZ2  #shape (1,10)


      dh1 = np.dot(dZ2, w2.T)  #shape (1,128)
      dz1 = dh1 * sigmoid_derivative(z1) #elementwise:(1,128)x(128,)=(1,128)

      temp4 = X[i].tolist()  #convert shape of X[i] from (784,) to (1, 784) - technical
      temp4 =np.array([temp4])

      dW1 = np.dot(temp4.T, dz1) # shape of (784,128)
      db1 = dz1 # shape of (1,128)

      if i == 0 and epoch == 0 :
        temp5 = b2.tolist()  #convert shape of b2 from (10,) to (1, 10) - technical
        b2 = np.array([temp5])

      if i == 0 and epoch == 0:
        temp6 = b1.tolist()  #convert shape of b1 from (128,) to (1, 128) - technical
        b1 = np.array([temp6])

      #  Update weights
      w2 -= learning_rate * dW2
      b2 -= learning_rate * db2
      w1 -= learning_rate * dW1
      b1 -= learning_rate * db1

    avg_epoch_loss = (avg_epoch_loss/train_size)
    print("Epoch:", epoch," Loss:", avg_epoch_loss)

### **3. Test the model and return the accuracy on the test set**

def test(X, y):
  true_pred_counter = 0
  Xtest_size = len(X)

  #predict labels for all x in X
  for i in range(Xtest_size):
    z1 = np.dot(X[i],w1) + b1
    h1 = sigmoid(z1)
    Z2 = np.dot(h1,w2) + b2
    y_hat = softmax(Z2)

    #if argmax in y_pred == true label -> true_counter++
    y_pred_max_index = np.argmax(y_hat, axis=1)
    y_pred_max_index = str(y_pred_max_index[0])
    if y_pred_max_index == y[i]:
      true_pred_counter += 1

  #divide counter in #data
  accuracy = true_pred_counter / Xtest_size

  return accuracy

"""### **4. Main**"""

from google.colab import drive
drive.mount('/content/drive')

train(X_train, y_train, 10)
accuracy = test(X_test, y_test)

print(accuracy)
