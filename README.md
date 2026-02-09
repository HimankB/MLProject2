# German Traffic Sign Classification

A machine learning pipeline for classifying German traffic signs using deep learning and classical algorithms.

## Project Overview

This project implements and evaluates multiple machine learning models for classifying 43 distinct classes of German traffic signs. The primary focus is on developing a robust Convolutional Neural Network (CNN) that learns directly from image data, complemented by classical baseline models for comparison.

### Key Objectives

- Design and implement a complete ML pipeline for traffic sign recognition
- Handle fine-grained visual distinctions between similar sign classes
- Compare performance across different model architectures
- Analyze trade-offs in feature representation, training complexity, and predictive confidence

---

## Models Implemented

### 1. Convolutional Neural Network (CNN) ⭐ Primary Model

**Architecture**
- Input: 64×64 RGB images, normalized to [0, 1]
- 3 convolutional blocks with increasing filter counts (32 → 64 → 128)
- Each block contains:
  - 2× Conv2D layers
  - Batch Normalization
  - ReLU activation
  - Max Pooling (2×2)
  - Dropout regularization
- 2 fully connected layers with dropout (0.5)
- Softmax output layer (43 classes)

**Training Configuration**
- Optimizer: Adam (learning rate = 0.001)
- Loss function: Sparse Categorical Crossentropy
- Callbacks: Early stopping, learning rate reduction on plateau, model checkpointing
- Data augmentation: Rotations, translations, zoom, shearing (no horizontal flips to preserve left/right semantics)

**Performance**
- **Validation Accuracy:** 97.52%
- **Test Accuracy:** ~99%
- **Average Prediction Confidence:** 97.5%
- **High Confidence Predictions:** 2,200+ out of 2,353 predictions above 90% confidence
- **Low Confidence Predictions:** Only 22 predictions below 50% confidence

**Strengths**
- Learns hierarchical features automatically from raw pixels
- Excellent generalization across all 43 classes
- Robust to real-world distortions through data augmentation
- Stable training convergence (7% → 96% accuracy over 28 epochs)

**Error Analysis**
- Misclassifications primarily between visually similar classes (e.g., speed limit signs with different numbers)
- Struggles with motion blur and partial occlusions
- Fine-grained intra-class distinctions remain challenging
- Performance highest on large, distinct signs; lowest on signs with subtle visual cues

---

### 2. Random Forest Classifier (RF)

**Methodology**
- Features: Histogram of Oriented Gradients (HoG) + PCA
- Configuration: 100 trees (n_estimators=100)
- Preprocessing: Feature standardization

**Performance**
- **Validation Accuracy:** ~57%

**Analysis**
- Struggled with visual subtleties in similar sign classes
- HoG + PCA features lost critical spatial information
- Surface-level feature reliance led to poor generalization
- Serves as a classical baseline to highlight CNN advantages

---

### 3. Multi-Layer Perceptron (MLP)

**Methodology**
- Input: Flattened HoG + PCA features
- Standard feedforward architecture

**Performance**
- **Validation Accuracy:** ~60%

**Analysis**
- Outperformed Random Forest but significantly weaker than CNN
- Loss of spatial structure from flattened features
- No ability to recognize 2D patterns (edges, corners, symmetry)
- Confirms importance of spatial feature representation

---

### 4. Support Vector Machine (SVM)

**Status:** Excluded from final analysis

**Reason**
- Poor training speed
- Consistently low performance
- Less interpretable than MLP
- Outperformed by all other models

---

## Results Summary

| Model | Validation Accuracy | Key Limitation |
|-------|---------------------|----------------|
| **CNN** | **97.52%** | Fine-grained intra-class distinctions |
| MLP | ~60% | Lost spatial structure |
| Random Forest | ~57% | Surface-level features only |
| SVM | Not reported | Poor speed and performance |

---

## Key Findings

### Why CNN Outperforms Classical Models

1. **Automatic Feature Learning:** CNNs learn hierarchical patterns from raw pixels without manual feature engineering
2. **Spatial Awareness:** Convolutional layers preserve 2D structure, critical for visual recognition
3. **Regularization:** Batch normalization and dropout prevent overfitting despite model depth
4. **Data Augmentation:** Simulates real-world variations, improving robustness

### Classical Model Limitations

- **Feature Engineering Bottleneck:** HoG + PCA may miss critical spatial cues
- **No Spatial Priors:** MLPs and RFs treat features as independent, ignoring 2D relationships
- **High-Dimensional Challenges:** Struggle with complex, multi-class image data

---

## Design Justifications

### CNN Architecture Choices

- **3-Block Structure:** Balances model complexity with training stability
- **Increasing Filter Counts (32→64→128):** Captures progressively abstract features
- **Batch Normalization:** Improves convergence and stabilizes gradients
- **Dropout (0.5 before FC layers):** Prevents overfitting in dense representations
- **Adam Optimizer (lr=0.001):** Fast convergence with smooth learning
- **No Horizontal Flips:** Preserves directional semantics of traffic signs

Early experiments with shallower models or without batch normalization suffered from overfitting or vanishing gradients.

---

## Future Improvements

1. **Transfer Learning:** Leverage pre-trained models (e.g., ResNet, EfficientNet)
2. **Attention Mechanisms:** Focus on small but critical regions in images
3. **Higher Resolution Inputs:** Capture finer details for similar classes
4. **Model Ensembling:** Combine predictions from multiple architectures
5. **Edge Deployment:** Model quantization and compression for real-time inference

---

## Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**
- 43 distinct traffic sign classes
- Varying lighting conditions, angles, and occlusions
- Resized to 64×64 pixels for training

---

## Installation & Usage

### Requirements
```bash
pip install tensorflow keras scikit-learn numpy pandas matplotlib
```

### Training the CNN
```python
# Preprocess images
images = preprocess_images(train_data, target_size=(64, 64))

# Train model
model = build_cnn_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_data, epochs=30, callbacks=[early_stopping, lr_scheduler])
```

### Making Predictions
```python
predictions = model.predict(test_images)
confidence = np.max(predictions, axis=1)
```

---

## References

- University of Melbourne lectures and notes (MLP, Random Forest concepts)
- TensorFlow CNN Tutorial: https://www.tensorflow.org/tutorials/images/cnn
- ChatGPT for detailed concept explanations (MLP, RF, SVM, TensorFlow/Keras CNN)
- Khan Academy (MLP and Random Forest videos)
- YouTube tutorials for deep learning model understanding

---

## License

This project is for educational purposes.

---

## Author

Machine Learning Project - Traffic Sign Classification

**Date:** 2024

**Acknowledgments:** Thanks to the creators of the GTSRB dataset and the open-source ML community.
