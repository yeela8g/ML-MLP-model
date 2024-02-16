## Neural Network using NumPy

This Python program implements a multi-layer perceptron (MLP) for handwriting recognition using the MNIST dataset. Below is a breakdown of the program's structure and functionalities.

### 1. Dataset

- The MNIST dataset is fetched using `fetch_openml` from the `sklearn.datasets` module.
- The dataset consists of 70,000 samples with 784 features (28x28 pixels) and 10 classes (digits 0-9).
- Data normalization is performed using Min-Max normalization.

### 2. Training

- The program implements forward and backward propagation for training the neural network.
- Activation functions such as sigmoid and softmax are implemented.
- Negative Log Likelihood loss function is used to compute the loss.
- Parameters (weights and biases) are initialized randomly.
- The training function iterates over the dataset for a specified number of epochs and updates the weights accordingly.
- During training, the loss decreases over epochs, and this is displayed for each epoch.

### 3. Testing

- The trained model is tested on the test set to evaluate its accuracy.
- Predictions are made on the test set, and accuracy is calculated by comparing predicted labels with true labels.
- The final accuracy on the test set is printed at the end of execution.

### Running the Program

- The program is designed to run in a Python environment that supports the required libraries.
- Make sure to mount your Google Drive to access the dataset.
- Execute the provided code, which trains the neural network and evaluates its performance on the test set.
- The program may run slow, taking about 10 minutes to complete due to the number of epochs and the size of the dataset.
- The loss decreases during the epochs is displayed, providing insights into the training process.
- The final accuracy on the test set is printed at the end of execution.

  ![image](https://github.com/yeela8g/ML-MLP-model/assets/118124478/4279458b-33d6-4e4c-a9f2-5bbd41ca16c2)

