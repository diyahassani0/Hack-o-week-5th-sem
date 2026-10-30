# Iris Flower Classification Using Neural Networks from Scratch

## Project Overview

This project implements a simple Artificial Neural Network (ANN) from scratch using NumPy to classify Iris flowers into three species: Setosa, Versicolor, and Virginica.

The main objective is to understand the working of neural networks, including perceptrons, activation functions, forward propagation, backpropagation, loss functions, and optimization algorithms. The project also demonstrates how to save and load trained models using `pickle` and `joblib`.

The entire implementation is performed in **Python using Google Colab**.

## Objectives

- Understand the working of a perceptron and artificial neurons.
- Implement activation functions such as ReLU, Sigmoid, and Softmax.
- Understand forward propagation and backpropagation.
- Calculate loss using categorical cross-entropy.
- Implement and compare SGD and Adam optimizers.
- Build and train a neural network using NumPy without TensorFlow or PyTorch.
- Evaluate the model using accuracy, a classification report, and a confusion matrix.
- Save and load trained models using `pickle` and `joblib`.

## Dataset

**Dataset:** Iris Flower Dataset

The Iris dataset contains 150 flower samples with four numerical input features.

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal in centimeters |
| Sepal Width | Width of the sepal in centimeters |
| Petal Length | Length of the petal in centimeters |
| Petal Width | Width of the petal in centimeters |

**Target classes:**
- Iris Setosa
- Iris Versicolor
- Iris Virginica

The dataset is loaded from the uploaded `IRIS.csv` file in Google Colab. The code identifies numerical features and attempts to identify the target column automatically.

## Technologies Used

- **Python** — Main programming language
- **NumPy** — Neural network calculations and matrix operations
- **Pandas** — Dataset loading and preprocessing
- **Matplotlib** — Graphs and visualizations
- **Scikit-learn** — Dataset splitting and model evaluation metrics
- **Pickle** — Saving and loading the trained model
- **Joblib** — Alternative model serialization method
- **Google Colab** — Execution environment

## Neural Network Architecture

The neural network contains three computational stages:

1. **Input Layer:** Receives the four flower measurements.
2. **Hidden Layer:** Contains 10 neurons using the ReLU activation function.
3. **Output Layer:** Contains three output neurons using Softmax to predict the flower species.

The network learns by repeatedly predicting the output, calculating the error, propagating gradients backward, and updating its weights.

## Algorithms and Concepts Implemented

### 1. Perceptron

A perceptron calculates a weighted sum of inputs, adds a bias, and applies an activation function.

### 2. Activation Functions

- **ReLU:** Introduces non-linearity in the hidden layer.
- **Sigmoid:** Demonstrates probability-based activation for binary classification.
- **Softmax:** Converts output scores into probabilities for three-class classification.

### 3. Forward Propagation

Passes input features through the network to calculate predicted class probabilities.

### 4. Loss Function

Categorical cross-entropy measures the difference between the predicted probabilities and the correct class labels.

### 5. Backpropagation

Calculates gradients of the loss with respect to the weights and biases so that the model can learn from prediction errors.

### 6. Optimization Algorithms

- **Stochastic Gradient Descent (SGD):** Updates parameters using calculated gradients.
- **Adam:** Uses moving averages of gradients and squared gradients to adapt parameter updates.

Both optimizers are trained and evaluated to compare their performance.

## Workflow

1. Upload the `IRIS.csv` dataset.
2. Load and inspect the dataset.
3. Identify input features and the target column.
4. Split the data into training and testing sets.
5. Normalize the input features.
6. Encode the target classes using one-hot encoding.
7. Initialize the neural network parameters.
8. Perform forward propagation and calculate loss.
9. Implement backpropagation.
10. Train the model using SGD and Adam.
11. Compare training loss and test accuracy using graphs.
12. Evaluate predictions using a classification report and confusion matrix.
13. Save and reload the trained model using `pickle` and `joblib`.

## Graphs and Evaluation

The project generates the following visualizations:

- **Dataset Visualization:** Shows the relationship between selected flower measurements and species.
- **Loss Comparison Graph:** Compares how training loss changes over epochs for SGD and Adam.
- **Accuracy Comparison Graph:** Compares test accuracy for both optimizers.
- **Confusion Matrix:** Shows correct and incorrect predictions for each flower species.

The actual accuracy and loss values depend on the training configuration and should be taken from the executed notebook.

## Model Deployment and Serialization

The trained model is saved in two formats:

- `iris_neural_network.pkl`
- `iris_neural_network.joblib`

Both files contain the trained weights, biases, normalization values, and class information needed to make predictions later without retraining the model.

The saved model can be loaded to predict the species of a new flower using its four measurements.

**Security note:** Only load Pickle or Joblib files from trusted sources.

## How to Run the Project

1. Download or clone this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Upload the `.ipynb` notebook.
4. Run the notebook cells in order.
5. Upload `IRIS.csv` when prompted.
6. Review the generated graphs and evaluation metrics.
7. Download the `.pkl` and `.joblib` files generated by the notebook.

### Required Libraries

The notebook uses NumPy, Pandas, Matplotlib, Scikit-learn, and Joblib. These libraries are generally available in Google Colab.

If running locally, install them using:

```bash
pip install numpy pandas matplotlib scikit-learn joblib
```

## Expected Outcome

By completing this project, students will understand the fundamental operations of a neural network and how a model learns from data. They will also gain practical experience comparing optimization algorithms, evaluating classification performance, and saving trained models for future use.

## Conclusion

This project demonstrates the complete basic neural network workflow, from data preprocessing and model training to evaluation and model persistence. Implementing the network using NumPy helps explain the mathematics behind forward propagation, backpropagation, and gradient-based optimization.

The project provides a foundation for understanding more advanced deep learning frameworks and model deployment techniques.

## Author

Student Machine Learning Assignment — Neural Networks and Model Deployment Basics
