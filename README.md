# Digit Recognizer with TensorFlow

This project uses a Convolutional Neural Network (CNN) model built with TensorFlow to recognize handwritten digits. The model is trained and validated on the MNIST dataset, which is a large database of handwritten digits commonly used for training and testing in the field of machine learning.

## Project Structure

The project is structured as follows:

1. **Data Loading and Preprocessing:** The train and test data are loaded and preprocessed. The images are normalized and reshaped to the appropriate shape for the CNN. The labels are one-hot-encoded.

2. **Data Augmentation:** The training data is augmented to increase the diversity of the data available for model training, without actually collecting new data. This includes rotations, shifts, and zooms.

3. **Model Building:** The CNN model is built using Keras' Sequential API. It includes several convolutional layers, max pooling layers, dense layers, and dropout layers.

4. **Model Training:** The model is trained for 20 epochs using the Adam optimizer and categorical cross-entropy as the loss function. Callbacks for learning rate reduction and model checkpointing are used.

5. **Model Evaluation:** The model's accuracy and loss are plotted for both the training and validation sets to assess the model's performance.

## Running the Model

To run the model, simply execute the Python script. The script will train the model and save the predictions for the test dataset in a file named `submission.csv`. This file contains two columns: `ImageId` and `Label`, which represent the image id from the test dataset and the predicted digit respectively.

## Dependencies

This project uses the following libraries:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

## Setup

This application requires Python 3.6 or higher. All required libraries can be installed using next command:

```shell
pip install -r requirements.txt
```