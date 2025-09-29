

# Rice Leaf Disease Classification using Transfer Learning 🌾

This project develops a deep learning model to accurately classify common diseases in rice leaves from images. Built using TensorFlow and Keras, this end-to-end computer vision pipeline leverages transfer learning with the MobileNetV2 architecture to achieve high accuracy, aiding in the early detection and management of crop diseases.



## Key Features

  * **High-Accuracy Classification**: Classifies rice leaves into three disease categories (`Bacterialblight`, `Brownspot`, `Leafsmut`).
  * **Transfer Learning**: Built upon the powerful **MobileNetV2** model pre-trained on ImageNet, which allows for effective learning with a smaller dataset.
  * **End-to-End Pipeline**: Covers the complete workflow from data acquisition using the Kaggle API to model training, evaluation, and making predictions on new images.
  * **Data Augmentation**: Utilizes random flips and rotations to create a more robust model and prevent overfitting.
  * **Performance Visualization**: Generates plots for training & validation accuracy and loss to monitor the model's learning progress.



## 🛠️ Tech Stack

  * **Language**: Python
  * **Framework**: TensorFlow, Keras
  * **Core Libraries**: NumPy, Matplotlib
  * **Data Acquisition**: Kaggle API (`kagglehub`)
  * **Environment**: Google Colab



## 📋 Project Workflow

1.  **Data Acquisition**: The "Rice Leaf Disease Image" dataset is downloaded directly from Kaggle into the environment using the `kagglehub` Python library.
2.  **Data Preparation**:
      * Images are loaded into a `tf.data.Dataset` using `image_dataset_from_directory`, which automatically infers class labels from the folder structure.
      * The dataset is split into **training (80%)**, **validation (10%)**, and **testing (10%)** sets.
      * The data pipeline is optimized for performance using `.cache()` and `.prefetch()`.
3.  **Model Architecture**:
      * A **MobileNetV2** model is loaded as the base, with its pre-trained ImageNet weights.
      * The base model's layers are **frozen** to retain the learned features.
      * A new classification head is added on top, consisting of a `GlobalAveragePooling2D` layer, a `Dropout` layer for regularization, and a `Dense` layer with a `softmax` activation function for the final classification.
4.  **Training**: The model is compiled with the `Adam` optimizer and `SparseCategoricalCrossentropy` loss function. It's then trained for 10 epochs, monitoring both training and validation accuracy.
5.  **Evaluation**: The model's final performance is measured on the unseen test set to provide an unbiased accuracy score.
6.  **Prediction**: The trained model is used to make a prediction on a single, new image uploaded by the user.



## 📊 Results

The model achieved a final test accuracy of approximately **95-99%** (your exact accuracy may vary slightly). The training history shows a consistent increase in accuracy and decrease in loss, indicating effective learning.



## 🚀 How to Use

1.  **Clone the repository**:
   
    git clone https://github.com/your-username/your-repo-name.git
   
2.  **Set up Kaggle API**: Upload your `kaggle.json` API key when prompted by the notebook.
3.  **Run the Notebook**: Open the `.ipynb` file in Google Colab or another Jupyter environment and run the cells in order.
4.  **Make a Prediction**: Follow the final steps in the notebook to upload your own image of a rice leaf and see the model's prediction.
