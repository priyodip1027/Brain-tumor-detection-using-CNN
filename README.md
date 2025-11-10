# Brain-tumor-detection-using-CNN
This project focuses on detecting brain tumors from MRI images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model classifies MRI scans into tumor and non-tumor categories, enabling early detection and analysis with high accuracy.

⚙️ Tech Stack

Language: Python

Frameworks: TensorFlow, Keras

Libraries:

NumPy, OpenCV, Matplotlib, scikit-learn, imutils

Model: Custom CNN architecture

Environment: Jupyter Notebook / VS Code

Hardware Support: GPU (CUDA enabled)

🧩 Features

✅ Detects brain tumors from MRI scan images
✅ Uses CNN layers like Conv2D, MaxPooling2D, and Dropout for robust feature extraction
✅ Includes Batch Normalization for stable training
✅ Implements callbacks like ModelCheckpoint, ReduceLROnPlateau, and TensorBoard for optimized performance
✅ Supports GPU acceleration (CUDA) for faster training
✅ Visualizes training and validation metrics using Matplotlib

🧠 Model Architecture

The model includes:

Multiple Convolutional + ReLU + MaxPooling layers for spatial feature extraction

Dropout layers to prevent overfitting

A Flatten + Dense network for classification

Softmax activation for binary classification (tumor / non-tumor)

📊 Dataset

The dataset contains MRI images of the human brain labeled as:

yes → contains tumor

no → healthy brain

(You can use publicly available datasets such as the Kaggle “Brain Tumor MRI Dataset.”)

🚀 Training

Data is preprocessed (resized, normalized, and shuffled).

Dataset is split into training and testing sets using train_test_split.

Model is trained with TensorBoard logging, learning rate reduction, and checkpoint saving.

📈 Results

High classification accuracy on test images

Model effectively learns spatial features of tumors

Visualization of loss and accuracy curves provided

Can be extended for multi-class classification (e.g., meningioma, glioma, pituitary tumor)
