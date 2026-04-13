#  EEG Mental Workload Classification

This project develops a machine learning system to classify **mental workload** using EEG (Electroencephalography) data. It compares a **Logistic Regression model built from scratch** with a **Multi-Layer Perceptron (MLP)** to understand how different models capture cognitive patterns in brain signals.

---

##  Project Overview

Mental workload reflects the cognitive demand placed on an individual during task performance. Accurate classification of workload is critical in areas such as:

* Aviation safety
* Medical decision-making
* Human-computer interaction

This project uses EEG signals to classify workload into **high vs low states**, leveraging both classical machine learning and deep learning approaches.

---

##  Dataset


Due to dataset size and access restrictions, a sample dataset is provided.

- `X_sample.npy` → feature matrix (50 samples, 620 features)
- `Y_sample.npy` → labels

To reproduce full results, use the complete dataset.


---

##  Feature Engineering

To reduce dimensionality and improve model performance, a feature extraction pipeline was applied:

###  Steps:

1. **Fourier Transform (FFT)**
   Converted EEG signals from time domain → frequency domain

2. **Frequency Band Extraction**
   Averaged spectral power into 5 standard EEG bands:

| Band  | Frequency Range (Hz) |
| ----- | -------------------- |
| Delta | 0.5 – 4              |
| Theta | 4 – 8                |
| Alpha | 8 – 13               |
| Beta  | 13 – 30              |
| Gamma | 30 – 50              |

3. **Feature Construction**

* 5 features per second × 2 seconds = 10 features per electrode
* 62 electrodes → **620 features per sample**

Final dataset shape:

X: (360 samples, 620 features)

---

##  Models Implemented

###  Logistic Regression (From Scratch)

Built using only NumPy:

* Forward propagation:
  z = WᵀX + b
* Activation: Sigmoid
* Loss: Binary Cross-Entropy
* Optimization: Gradient Descent

✔ Purpose: baseline model + deep understanding of ML fundamentals

---

###  Multi-Layer Perceptron (MLP)

Implemented using Keras & TensorFlow:

* Input layer: 620 features
* Hidden layers:

  * Dense (64 neurons, ReLU)
  * Dense (32 neurons, ReLU)
* Output layer:

  * 2 neurons (Softmax)

✔ Captures **non-linear relationships** in EEG data

---

##  Model Evaluation

* **5-Fold Cross-Validation**
* Fixed random seed for reproducibility
* Metric: **Classification Accuracy**

---

##  Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 88.61%   |
| MLP                 | 94.17%   |

---


##  Key Insights

* MLP significantly outperforms Logistic Regression
* EEG workload patterns are **non-linear**
* Feature engineering was critical for performance
* Dropout **reduced performance** due to small dataset size
* Proper scaling and preprocessing improved convergence

---

##  Hyperparameter Tuning Insights

| Learning Rate | Dropout | Accuracy   |
| ------------- | ------- | ---------- |
| 0.001         | 0.3     | 81.11%     |
| 0.01          | 0.3     | 69.72%     |
| 0.001         | 0.2     | 79.72%     |
| 0.001         | 0.0     | **94.17%** |

✔ Best performance achieved **without dropout**

---

##  Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras
* Jupyter Notebook

---

##  How to Run

```bash
git clone https://github.com/adedolapo-adeniran/eeg-mental-workload-classification.git
cd eeg-mental-workload-classification

pip install -r requirements.txt

jupyter notebook
```

---

##  Project Structure

```
eeg-mental-workload-classification/
│
├── README.md
├── requirements.txt
├── notebooks/
├── src/
├── visuals/
└── report/
```

---


## Future Improvements

* Implement Convolutional Neural Networks (CNNs)
* Explore time-series deep learning models (LSTM, Transformers)
* Increase dataset size for better generalization
* Apply advanced feature extraction techniques

---

##  Acknowledgements

This project was developed as part of the **CE889: Neural Networks and Deep Learning** module.

---

##  If you found this useful

Give the repo a ⭐ and feel free to connect!

## Author

Adedolapo Adeniran
