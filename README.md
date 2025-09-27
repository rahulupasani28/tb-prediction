# TB-prediction
Tuberculosis (TB) Prediction Model 
This repository has a deep learning–driven Tuberculosis (TB) Prediction Model based on chest
X-ray images. The project utilizes Convolutional Neural Networks (CNNs) and sophisticated
preprocessing methods for classifying X-ray images as TB-positive or TB-negative, enabling quicker
and more accurate diagnostic procedures.

Features - Chest X-ray image preprocessing - Deep learning–driven classification
(TensorFlow/Keras) - Training and evaluation scripts - Streamlit-based web application for
predictions - Visualization of predictions with probability scores

requirements.txt - Dependencies README.md - Project documentation

Dataset 
This model is trained on Chest X-ray datasets for Tuberculosis detection. If you would like
to duplicate the results, you can download openly available TB datasets like:
- TBX11K
Dataset (https://arxiv.org/abs/1903.09820)
- Shenzhen & Montgomery TB Datasets
(https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MontgomeryCountyChestXraySet.html)
Methodology
1. Image Preprocessing – Resizing, normalization, and augmentation
2. Deep Learning Model – CNN structure with focal loss for imbalanced data
3. Evaluation – Accuracy, precision, recall, F1-score, and AUC metrics 4. Visualization – Heatmaps/Grad-CAM for
explainability

 In case you use this work or dataset, kindly cite the following reference:
Tawsifur Rahman, Amith Khandakar, Muhammad A. Kadir, Khandaker R. Islam, Khandaker F.
Islam, Zaid B. Mahbub, Mohamed Arselene Ayari, Muhammad E. H. Chowdhury. (2020). Reliable
Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization.
IEEE Access, Vol. 8, pp. 191586–191601. DOI: 10.1109/ACCESS.2020.3031384
