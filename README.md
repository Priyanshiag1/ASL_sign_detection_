# 🧠 ASL Sign Language Recognition using Deep CNN  
A deep learning project to classify **American Sign Language (ASL)** letters using a **Convolutional Neural Network (CNN)** trained on the **Sign Language MNIST** dataset.

---

## 📁 Dataset  
- **Dataset Used**: [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist)  
- 28x28 grayscale images representing ASL signs for **24 alphabets** (A-Y, except J,Z)

---

## 🧠 Model Architecture

| Layer Type          | Output Shape        | Details                              |
|---------------------|---------------------|---------------------------------------|
| Input               | (28, 28, 1)         | Grayscale ASL image                   |
| Conv2D (32 filters) | (26, 26, 32)        | Kernel = 3x3, ReLU                    |
| BatchNormalization  | (26, 26, 32)        | -                                     |
| MaxPooling2D        | (13, 13, 32)        | Pool size = 2x2                       |
| Conv2D (64 filters) | (11, 11, 64)        | Kernel = 3x3, ReLU                    |
| BatchNormalization  | (11, 11, 64)        | -                                     |
| MaxPooling2D        | (5, 5, 64)          | Pool size = 2x2                       |
| Conv2D (128 filters)| (3, 3, 128)         | Kernel = 3x3, ReLU                    |
| BatchNormalization  | (3, 3, 128)         | -                                     |
| MaxPooling2D        | (1, 1, 128)         | Pool size = 2x2                       |
| Flatten             | 128                 | -                                     |
| Dense               | 128                 | ReLU activation                       |
| Dropout             | 0.5                 | Prevents overfitting                  |
| Dense               | 25                  | Softmax output for 25 classes         |

---

## 🧪 Training Strategy
- **Optimizer**: Adam (lr = 0.001)  
- **Loss**: Categorical Crossentropy  
- **Metrics**: Accuracy  
- **Regularization**:  
  - Dropout (0.5)  
  - Data Augmentation (rotation, zoom, shift)  
  - Early Stopping (patience=3)

---

## 📊 Results

| Metric              | Value              |
|---------------------|--------------------|
| Final Train Acc     | ~98.8%             |
| Final Val Acc       | ~99.7%             |
| Final Val Loss      | ~0.0060            |

⚠️ A version without data augmentation achieved slightly lower validation loss (~0.0014), but showed signs of overfitting. This model was selected for better **generalization**.

---
