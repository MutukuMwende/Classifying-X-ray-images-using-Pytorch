# Classifying-X-ray-images-using-Pytorch
# Pneumonia Classification Using Deep Learning

## Overview

Pneumonia is a significant respiratory illness and one of the leading causes of death worldwide, particularly in low-resource settings. Early and accurate diagnosis is critical for effective treatment. Chest X-rays (CXRs) are commonly used for pneumonia diagnosis, but the manual review process can be time-consuming and prone to human error.

This project leverages a **pre-trained ResNet-18 deep learning model** to classify chest X-ray images into two categories: 
- **NORMAL**: Healthy lungs
- **PNEUMONIA**: Lungs affected by pneumonia.

By fine-tuning ResNet-18's parameters, we aim to develop an efficient and accurate classification model that assists in clinical decision-making.  

---

## Project Objective

- **Main Goal**: Use transfer learning with ResNet-18 to classify X-ray images into **normal** or **pneumonia-affected** lungs.  
- **Model Performance Metrics**: 
  - Accuracy: Measures the proportion of correct predictions.
  - F1-Score: Balances precision and recall, especially useful for imbalanced datasets.

---

## Methodology

1. **Dataset**:  
   - The dataset contains preprocessed X-ray images divided into:
     - **Training Set**: 150 images each for NORMAL and PNEUMONIA (300 in total).
     - **Test Set**: 50 images each for NORMAL and PNEUMONIA (100 in total).
   - Images are normalized using the ResNet-18 standard normalization parameters.

2. **Pre-trained Model**:  
   - **ResNet-18**: A convolutional neural network with pre-trained weights.
   - The final fully connected layer was modified to output a single value for binary classification.

3. **Training**:  
   - Used Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`) for binary classification.
   - Optimized only the final layer using Adam optimizer with a learning rate of 0.01.
   - Fine-tuned the model for **3 epochs**.

4. **Evaluation**:  
   - **Metrics**: Accuracy and F1-Score were used to evaluate performance on the test set.
   - Predicted probabilities were converted to binary outputs using a threshold of 0.5.

---

## Results

### Training Performance
The training accuracy and loss plateaued after three epochs:
- **Final Training Accuracy**: ~50.33%
- **Final Training Loss**: ~0.9199

### Test Performance
- **Test Accuracy**: 58.0%
- **Test F1-Score**: 70.4%

---

## Significance of the Results

### Accuracy:
The accuracy of **58%** indicates that the model correctly identified slightly over half of the images as normal or pneumonia-affected. While this result is not ideal, it reflects the model's current limitations and suggests the need for further fine-tuning or more diverse training data.

### F1-Score:
The F1-score of **70.4%** shows a relatively better balance between precision (how many predicted pneumonia cases were correct) and recall (how many actual pneumonia cases were identified). This metric is particularly relevant for medical diagnosis, where both false positives and false negatives carry significant risks.

---

## Challenges and Future Improvements

### Challenges
- **Limited Dataset**: With only 300 training images, the model had insufficient data to generalize effectively.
- **Early Stopping**: Training for only three epochs may not have allowed the model to reach its full potential.

### Improvements
- **Dataset Augmentation**: Increase training data by using image augmentation techniques like rotation, flipping, and zooming.
- **Extended Training**: Train the model for more epochs while monitoring overfitting.
- **Fine-Tuning Entire Network**: Allow more layers of ResNet-18 to update their weights instead of freezing them.
- **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and optimizers.
- **Advanced Models**: Explore more complex architectures like ResNet-50 or DenseNet.

---

## Conclusion

This project demonstrates the potential of transfer learning in classifying medical images, even with limited computational resources. While the current performance metrics indicate room for improvement, the approach lays a foundation for more robust pneumonia detection systems. With further fine-tuning and enhancements, AI-driven diagnostic tools like this one can play a critical role in improving healthcare delivery, particularly in resource-limited settings.
