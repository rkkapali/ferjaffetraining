# ferjaffetraining

Facial Emotion Recognition Using a Custom CNN on FER-2013 and JAFFE Datasets
Overview
This project develops a facial emotion recognition (FER) system using a custom Convolutional Neural Network (CNN) to classify six emotions: neutral, angry, happy, surprise, sad, and fear. The model is trained on the FER-2013 dataset and fine-tuned on the JAFFE dataset, achieving test accuracies of 49.40% on FER-2013 and 89.19% on JAFFE.

Datasets
FER-2013: A large dataset of ~35,000 grayscale images (48x48 pixels) with real-world diversity, sourced from Kaggle (Msambare, 2020).
Link: FER-2013 on Kaggle
JAFFE: A smaller dataset of ~213 grayscale images from Japanese female subjects, captured under controlled conditions (Lyons et al., 1998).
Stored in /content/drive/MyDrive/JAFFE-[70,30]/JAFFE-[70,30].
Model
Initial Approach: Used ResNet-50 with transfer learning, but achieved low accuracies (25.1% on FER-2013, 21.6% on JAFFE).
Final Model: A custom CNN with 5 convolutional layers (32, 64, 128, 256, 256 filters), BatchNormalization, MaxPooling, and a dense layer with dropout (0.4), totaling ~1.28M parameters.
Training:
FER-2013: 20 epochs, batch size 32, with data augmentation (rotations, shifts, zooms, flips).
JAFFE: Fine-tuned for 20 epochs, batch size 32, no augmentation.
Optimizer: Adam (learning rate 0.0005), EarlyStopping (patience=5).
Results
FER-2013 Test Accuracy: 49.40%
JAFFE Test Accuracy: 89.19%
Visuals (in report):
Accuracy/loss plots for FER-2013 and JAFFE.
Confusion matrices and sample predictions for both datasets.
Requirements
Python 3.11
Libraries: tensorflow, numpy, opencv-python, matplotlib, seaborn, scikit-learn, kagglehub
Google Colab with GPU (recommended)
Setup and Running
Mount Google Drive:
Ensure JAFFE dataset is in /content/drive/MyDrive/JAFFE-[70,30]/JAFFE-[70,30].
