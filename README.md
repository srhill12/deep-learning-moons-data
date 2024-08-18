
# Nonlinear Data Classification with Neural Network

This repository contains a Jupyter notebook that demonstrates the process of building and evaluating a neural network model for classifying nonlinear data. The dataset used in this project is a synthetic moons dataset, which is commonly used for binary classification tasks.

## Project Overview

In this project, we:

- Import and preprocess the nonlinear moons dataset.
- Build and train a neural network using TensorFlow and Keras.
- Evaluate the model's performance on both training and test data.
- Analyze the accuracy and loss of the model over 200 epochs.

## Steps

### 1. Data Import and Preprocessing

- The dataset is imported from a CSV file. It contains two features (`X1`, `X2`) and a target column (`Target`), which indicates the class labels.
- We split the data into training and testing sets using an 80/20 split ratio.
- A `StandardScaler` is used to normalize the feature data.

### 2. Neural Network Model

- We create a Keras Sequential model with the following layers:
  - **Input Layer:** Fully connected layer with 6 units and ReLU activation.
  - **Hidden Layer:** Fully connected layer with 6 units and ReLU activation.
  - **Output Layer:** Fully connected layer with 1 unit and sigmoid activation.
- The model is compiled using binary cross-entropy as the loss function and the Adam optimizer.
- The model is trained over 200 epochs.

### 3. Model Evaluation

- We evaluate the model on the test data, achieving a final accuracy of 100% with a very low loss value.
- The training accuracy and loss are tracked and printed for each epoch to analyze the model's learning process.

## Results

- **Model Accuracy:** 100% on the test dataset.
- **Model Loss:** 0.000694 on the test dataset.
  
The model performs exceptionally well on this dataset, achieving perfect accuracy on the test data.

## Requirements

To run this project, the following Python libraries are required:

- pandas
- matplotlib
- scikit-learn
- tensorflow

You can install these dependencies using pip:

```bash
pip install pandas matplotlib scikit-learn tensorflow
```

## Conclusion

This project showcases the effectiveness of a simple neural network in classifying nonlinear data. Despite its simplicity, the model achieves perfect accuracy on the given dataset, making it a strong example of the power of neural networks in binary classification tasks.
