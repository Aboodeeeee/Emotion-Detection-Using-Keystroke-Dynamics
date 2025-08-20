# Emotion Recognition using Keystroke Dynamics

This project explores **emotion recognition from keystroke dynamics** — analyzing typing behavior (latency, dwell time, flight time, etc.) to infer a user’s emotional state. The goal is to build machine learning models that classify emotions such as *Happy, Sad, Angry, Neutral*, etc., using keystroke timing data.

---

## 📌 Project Overview
- **Type of Problem:** Supervised Learning (Classification)
- **Dataset:** [EmoSurv Dataset](https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using)  
- **Features:**  
  - **Dwell Time:** Time a key is held down  
  - **Flight Time:** Interval between successive key presses  
  - **Latency:** Delay between release and next press  
  - **Typing Speed:** Overall speed of typing  
  - **Edit Distance (typos):** Newly introduced feature to capture typing errors  

---

## ⚙️ Preprocessing
- Standardized categorical features (participants info).  
- Converted numeric fields, handled missing values, normalized ranges.  
- Applied **Min-Max Scaling** to bring all features to `[0,1]`.  
- Encoded categorical features with `LabelEncoder`.  

---

## 🧠 Models Implemented
We experimented with four classifiers using **10-fold cross-validation**:
- Random Forest (RF)  
- XGBoost (XGB)  
- Support Vector Machine (SVM)  
- Multi-Layer Perceptron (MLP)  

---

## 📊 Results

### Free Text Dataset
| Model | F1 Micro (±std) | F1 Macro (±std) |
|-------|----------------|-----------------|
| RF    | 0.211 ± 0.051  | 0.199 ± 0.061   |
| XGB   | **0.451 ± 0.050** | 0.284 ± 0.075   |
| SVM   | 0.472 ± 0.013  | 0.128 ± 0.002   |
| MLP   | 0.312 ± 0.107  | 0.183 ± 0.074   |

👉 **XGBoost achieved the most balanced performance** across both micro and macro F1.  
👉 SVM achieved the highest micro-F1 but suffered in macro-F1 (class imbalance).  

### Fixed Text Dataset
- Accuracy: **67%**  
- F1 Micro: **0.39**  
- F1 Macro: **0.15**  
- Precision: **0.11**  
- F1 Score: **0.13**  

👉 Controlled text input improved consistency but reduced generalization compared to free text.  

### Overall Best Model: **XGBoost**
- Accuracy: **80%**  
- Precision: **41%**  
- F1 Micro: **0.49**  
- F1 Macro: **0.37**  

---

## 🔑 Key Contributions
- Built a **keystroke-based emotion recognition model** achieving up to **80% accuracy** with XGBoost.  
- Introduced **edit distance (typo detection)** as a novel feature, improving classification.  
- Explored the impact of **demographics (age, gender, typing habits, education level)** on performance — some were useful, others added noise and required feature selection.  
- Provided a **comparative evaluation** of multiple ML models with metrics and confusion matrix analysis.  

---

## 🚀 Future Work
- Address class imbalance with **data augmentation / SMOTE**.  
- Explore **deep learning models** for richer temporal patterns.  
- Investigate **ensemble stacking** to improve robustness across datasets.  
- Extend to **real-time emotion recognition applications** (e.g., adaptive HCI, stress monitoring).  

---
