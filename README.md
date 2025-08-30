# custom-object-localization-with-tensorflow

# Object Localization with TensorFlow

This project demonstrates how to build and train a deep learning model for **object localization** using TensorFlow. The goal is to not only classify an object in an image but also to predict the bounding box coordinates around the object.

---

## 🎯 Project Overview

The notebook walks through the complete process of:

1. **Data Collection & Preprocessing**  
2. **Building a Convolutional Neural Network (CNN)**  
3. **Defining a Custom Loss Function** (combining classification and regression loss)  
4. **Training the Model**  
5. **Evaluating Performance**  
6. **Visualizing Predictions**

---

## 📁 Dataset

The project uses a custom dataset of emoji images (from [OpenMoji](https://openmoji.org/)) with annotated bounding boxes. Each image is labeled with:

- Class label (e.g., 😊)
- Bounding box coordinates: `(x_min, y_min, x_max, y_max)`

---

## 🧠 Model Architecture

The model is a **multi-output CNN** that predicts:

1. **Class probabilities** (softmax output)  
2. **Bounding box coordinates** (4 regression values)

### Example Architecture:

```python
model = tf.keras.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(72, 72, 3)),
    MaxPool2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(128, activation='relu'),
    # Two output layers:
    Dense(num_classes, activation='softmax', name='class_output'),
    Dense(4, name='bbox_output')  # [x_min, y_min, x_max, y_max]
])
```

---

## ⚙️ Custom Loss Function

The total loss is a weighted sum of:

- **Categorical cross-entropy** for classification  
- **Mean Squared Error (MSE)** for bounding box regression

```python
def custom_loss(y_true, y_pred):
    class_loss = tf.keras.losses.categorical_crossentropy(y_true[0], y_pred[0])
    bbox_loss = tf.keras.losses.mse(y_true[1], y_pred[1])
    return class_loss + 0.1 * bbox_loss  # weight regression less
```

---

## 📊 Training

The model is trained using:

- Optimizer: Adam
- Metrics: Accuracy (classification), IoU (localization)
- Callbacks: Early stopping, model checkpointing

---

## 📈 Evaluation

After training, the model is evaluated on a test set. Metrics include:

- Classification accuracy  
- Intersection over Union (IoU) for bounding boxes

---

## 👀 Visualization

Sample predictions are visualized with:

- Original image  
- Ground truth bounding box (green)  
- Predicted bounding box (red)  
- Class label and confidence

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python
   ```

2. **Download the dataset** (emoji images) and update the path in the notebook.

3. **Run the notebook** step-by-step to:
   - Load and preprocess data
   - Build and train the model
   - Evaluate and visualize results

---

## 📂 File Structure

```
Object_Localization_with_TensorFlow_Complete.ipynb  # Main notebook
emojis/                                            # Dataset folder
  - 1F469-1F3FC-200D-1F9B2.png
  - 1F97F.png
  - ...
```


## 🛠 Future Improvements

- Use a pre-trained backbone (e.g., ResNet, MobileNet)
- Implement data augmentation
- Try different loss functions (e.g., Smooth L1, CIoU)
- Deploy as a web app using TensorFlow.js or Flask

---
## 📜 License
The emoji images are from [OpenMoji](https://openmoji.org/) (CC BY-SA 4.0).  
---

## 👨‍💻 Author
Marwan Gamal

Built as part of the Coursera Guided Project:  
[Object Localization with TensorFlow](https://www.coursera.org/projects/object-localization-tensorflow)
