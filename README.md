🧠 Brain Tumor Classification using MobileNetV2
📄 Overview

This project is a Deep Learning-based Brain Tumor Classification System that predicts tumor type from MRI images.
It uses MobileNetV2 (a lightweight and efficient CNN model) with fine-tuning and data augmentation to classify brain MRI scans into one of four categories:

🧬 Glioma

🧠 Meningioma

💡 No Tumor

🩸 Pituitary Tumor

The trained model is deployed using Streamlit, providing an easy-to-use web interface for medical image analysis.

🚀 Features

✅ Deep Learning model trained on real MRI images
✅ MobileNetV2 for efficient feature extraction
✅ Image augmentation to improve generalization
✅ Streamlit web interface for real-time predictions
✅ Displays class probabilities with confidence scores
✅ Model saved in .h5 format for reuse

📂 Dataset

The dataset used consists of MRI brain images collected from open medical sources and preprocessed into four folders representing the classes.
Each image was resized to 224x224 pixels and normalized before feeding into the model.

🧩 Model Architecture

Base Model: MobileNetV2 (pretrained on ImageNet)

Top Layers: GlobalAveragePooling + Dense layers + Dropout

Loss Function: Focal Loss (for class imbalance)

Optimizer: Adam

Accuracy Achieved: ~31% (baseline — can be improved with more training data & fine-tuning)

🧠 How It Works

User uploads an MRI scan image.

Image is preprocessed and passed into the MobileNetV2 model.

Model predicts the tumor type with a confidence score.

Streamlit displays the result interactively.

🖥️ Tech Stack
Tool / Library	Purpose
Python	Programming language
TensorFlow / Keras	Deep learning framework
NumPy, PIL	Image processing


This project gave me hands-on experience in implementing Deep Learning for medical image classification using MobileNet architecture. I learned how to handle image preprocessing, data augmentation, and class imbalance using class weights. I also explored different loss functions such as Categorical Crossentropy and Focal Loss to handle class imbalance more effectively.

Through training and fine-tuning MobileNet, I understood how transfer learning, feature extraction, and fine-tuning deeper layers improve model performance. Deploying the trained model on Streamlit helped me connect the backend AI model with an interactive user interface for real-time brain tumor predictions.
Streamlit	Web app deployment
scikit-learn	Evaluation metrics
