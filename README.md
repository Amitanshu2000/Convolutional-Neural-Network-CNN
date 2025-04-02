# MNIST Handwritten Digit Recognizer using Deep CNN

## Project Overview
This project implements a Deep Convolutional Neural Network (CNN) model for recognizing handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras, following the LeNet-5 architecture with additional enhancements like dropout layers and data augmentation.

## Contributors
- Anirudh Gupta
- Amitanshu Tiwari

## Problem Statement
1. **Importance of Medical Image Analysis**: Essential for disease diagnosis, treatment planning, and healthcare automation.
2. **Challenges in Traditional Methods**: Inconsistencies, human errors, and delayed or inaccurate assessments.
3. **Project Goal**: Develop a Deep CNN model for automated medical image classification.
4. **Key Benefits**: Enhances diagnostic precision and reduces manual workload.
5. **Deep Learning Advantages**: Improves feature extraction and pattern recognition.
6. **Real-World Applications**: Tumor detection, pneumonia diagnosis, and skin lesion classification.

## Project Structure
1. **Importing Libraries**: TensorFlow, Keras, Pandas, and Matplotlib.
2. **Preparing the Dataset**: Loading and preprocessing the MNIST dataset.
3. **Model Building**: Implementing a Deep CNN based on LeNet-5 architecture.
4. **Model Fitting**: Training the model with GPU support.
5. **Model Analysis**: Evaluating performance using learning curves and confusion matrices.
6. **Predicting Using Test Data**: Generating predictions for the test dataset.

## Data Analysis
### 1. Importing the Libraries
- **TensorFlow v2**: An open-source machine learning framework from Google.
- **Keras**: A high-level neural network API running on top of TensorFlow.

### 2. Preparing the Dataset
- **Normalization**: Scaling pixel values from [0, 255] to [0, 1].
- **Reshaping**: Converting the dataset into a (28, 28, 1) matrix.
- **Encoding**: Converting labels into one-hot encoded vectors.
- **Train-Test Split**: Dividing the dataset into training and validation sets.

### 3. Model Building
- **Architecture**: Input → [[Conv2D → ReLU] × 2 → MaxPool2D → Dropout] × 2 → Flatten → Dense → Dropout → Output.
- **Data Augmentation**: Techniques like zooming, rotating, and flipping to expand the dataset.
- **Optimization Strategy**: Using RMSProp for faster convergence and ReduceLROnPlateau for dynamic learning rate adjustment.

### 4. Model Fitting
- **Training**: Conducted on Kaggle with GPU support for faster computation.
- **Monitoring**: Tracking training and validation losses to avoid overfitting.

### 5. Model Analysis
- **Learning Curve**: Shows decreasing training and validation losses.
- **Confusion Matrix**: Reveals model performance across digit classes.

### 6. Predicting Using Test Data
- **Predictions**: Stored in a CSV file for submission.

## Observations
- **Dataset**: Successfully loaded and preprocessed with balanced digit distribution.
- **Model**: Achieved high accuracy with some misclassifications.
- **Training**: Efficiently conducted using GPU support.

## Managerial Insights
- **Automation Potential**: Viable for automated data entry systems in banking and postal services.
- **Cost-Effectiveness**: Reduces labor costs associated with manual transcription.
- **Scalability & Adaptability**: Can be fine-tuned for different languages and scripts.
- **Performance vs. Infrastructure**: Requires computational resources but offers high accuracy.
- **Error Handling**: Active learning and periodic retraining can improve accuracy.

## Code Implementation
The Jupyter notebook includes detailed code for:
- Loading and preprocessing the MNIST dataset.
- Building and compiling the CNN model.
- Training the model with data augmentation.
- Evaluating performance and generating predictions.

## Usage
1. Clone the repository.
2. Install required libraries: `pip install tensorflow pandas numpy matplotlib seaborn`.
3. Run the Jupyter notebook to train the model and generate predictions.

## Results
The model demonstrates strong performance in classifying handwritten digits, making it suitable for real-world applications like digit recognition and form processing.
