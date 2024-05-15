# Malaria-Detection-Model-Using-TensorFlow
The project entails creating a model which is able to classify whether a blood smear is uninfected or not

In this endeavor, we showcase the development of a sophisticated deep learning architecture to discern whether cell images are afflicted with Malaria or not. Leveraging the power of TensorFlow 2 alongside the intuitive Keras API in Python, we construct a robust model. Trained on the Malaria Cell Images Dataset, our model excels in accurately discerning between infected and uninfected cells.
### Prerequisites

Before you begin, ensure you have the following libraries installed:

- NumPy
- TensorFlow
- OpenCV-Python
- scikit-learn
- Matplotlib

### Installation

1. Clone this repository:



2. Download the Malaria Cell Images Dataset from Kaggle and extract it https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

3. Place the `cell_images` folder in the project's working directory.

4. Create a `testing-samples` folder and move a few test images to it for later inference.

### Dataset

The dataset used in this project contains cell images categorized into two classes: Parasitized and Uninfected. These images are used for training and testing the model.

## Data Preprocessing

Images are preprocessed using OpenCV. They are converted to grayscale and resized to a (70x70) shape. The dataset is loaded, and images are normalized to help the neural network learn faster. Data is split into training and testing sets, and shuffling is performed.

## Model Architecture
The model architecture consists of convolution layers, activation functions, max-pooling layers, and fully connected layers. The output layer uses a sigmoid activation function for binary classification. Here's an overview of the model:
## Training

The model is trained for 3 epochs with a batch size of 64. You can adjust these hyperparameters according to your requirements. The training process will output accuracy and loss values.

## Model Evaluation

The model is evaluated on the testing dataset, and accuracy and loss metrics are displayed.

## Inference

You can use the trained model to make inferences on new images. Provided test images are used to demonstrate the model's ability to classify infected and uninfected cells.

## Saving the Model

The trained model can be saved for later use. You can load the model using the saved weights for quick inference without retraining.
