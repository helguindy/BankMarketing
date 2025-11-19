# Neural Network From Scratch – Bank Marketing Dataset

This project implements a complete **Neural Network from scratch** (no scikit-learn, no TensorFlow, no PyTorch).  
All computations; forward propagation, backpropagation, gradients, weight updates, activations, loss functions; are built manually to demonstrate how neural networks work internally.

The model is trained on the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

---

## 💸 Dataset

**Dataset Used:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

This dataset contains information about customers contacted during a marketing campaign conducted by a Portuguese bank.  
The goal is to predict whether a customer will **subscribe to a term deposit**.

---

## 💲 What the Dataset Includes

The dataset contains **45,211 rows** and **17 input features**, including:

### Client-related attributes:
- `age`
- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`

### Contact-related attributes:
- `contact` (cellular / telephone)
- `month` (of last contact)
- `day_of_week`
- `duration` (last contact duration)

### Campaign-related attributes:
- `campaign` (number of contacts performed)
- `pdays` (days since last contact)
- `previous` (number of previous contacts)
- `poutcome` (outcome of previous campaign)

### Social and economic attributes:
- `emp.var.rate`
- `cons.price.idx`
- `cons.conf.idx`
- `euribor3m`
- `nr.employed`

### Target label:
- `y` → `"yes"` or `"no"`  
  (whether the client subscribed to a term deposit)

---

## 🎯 Prediction Goal

The task is a **binary classification** problem:

- **1 → Yes**, the client subscribed  
- **0 → No**, the client did not subscribe  

The neural network outputs a probability using a **sigmoid activation**.

---

## 👑 **Real-world significance**

Banks invest heavily in marketing campaigns, and predicting which customers are likely to subscribe to a product is extremely valuable because it:

- Reduces campaign cost  
- Increases targeting efficiency  
- Improves customer engagement  
- Enhances decision-making through data-driven prediction  

Effective prediction models help banks avoid wasting resources on uninterested clients.

---


## 🧠 What Makes This Project Unique?

This project is built **completely from scratch**, meaning:

❌ No `sklearn.neural_network`  
❌ No `MLPClassifier`  
❌ No TensorFlow  
❌ No PyTorch  
❌ No built-in optimizers or training loops  

✔ All layers  
✔ All gradients  
✔ All activations  
✔ All forward/backward passes  
✔ All parameter updates  

…are coded manually.

This makes the project ideal for learning the true mechanics behind neural networks.

---

# 🧠 Neural Network Architecture

Network structure:  
```
[N, 10, 1]
```
Where:

- **N** = number of processed input features  
- **Hidden Layer:** 10 neurons  
- **Activation:** sigmoid (implemented manually)  
- **Output Layer:** 1 neuron with **sigmoid activation** (binary classification)


**Sigmoid (used in both hidden and output layers):**

$$\begin{aligned} A &= \frac{1}{1 + e^{-Z}} \\ 
A' &= A \cdot (1 - A) \end{aligned}$$


## 🚀 Forward Propagation

### **Hidden Layer**
$$\begin{aligned} Z_1 &= W_1 \cdot X + b_1 \\ A_1 &= \text{sigmoid}(Z_1) \end{aligned}$$

### **Output Layer**
$$\begin{aligned} Z_2 &= W_2 \cdot A_1 + b_2 \\ A_2 &= \text{sigmoid}(Z_2) \quad \text{(predicted probability)} \end{aligned}$$


## 🔁 Backpropagation 
We calculate how much each weight and bias should change to reduce the error.

#### **Output Layer**
$$dZ_2 = A_2 - Y \quad \text{(difference between predicted and true value)}$$

$$dW_2 = \frac{1}{m} \cdot dZ_2 \cdot A_1^T \quad \text{(gradient for output weights)}$$

$$db_2 = \frac{1}{m} \sum dZ_2 \quad \text{(gradient for output bias)}$$

#### **Hidden Layer**
$$dA_1 = W_2^T \cdot dZ_2 \quad \text{(error propagated to hidden layer)}$$

$$dZ_1 = dA_1 * A_1 * (1 - A_1) \quad \text{(adjust for sigmoid derivative)}$$

$$dW_1 = \frac{1}{m} \cdot dZ_1 \cdot X^T \quad \text{(gradient for hidden weights)}$$

$$db_1 = \frac{1}{m} \sum dZ_1 \quad \text{(gradient for hidden bias)}$$

#### **Parameter Updates**
$$W_1 := W_1 - \eta \cdot dW_1$$

$$b_1 := b_1 - \eta \cdot db_1$$

$$W_2 := W_2 - \eta \cdot dW_2$$

$$b_2 := b_2 - \eta \cdot db_2$$
  
--- 

# 🏋️  Training Procedure


### ✔ **Data splitting**
- **70% Training**
- **20% Validation**
- **10% Testing**

### ✔ **Batching**
Mini-batch size = **100 samples**

### ✔ **Epochs**
Minimum: **10 epochs**  

Training includes:
- Training loss per epoch  
- Training accuracy per epoch  
- Validation loss per epoch  
- Validation accuracy per epoch  

---

# 📉 Loss Function

#### **Binary Cross Entropy Loss (BCE)**

$$L = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \cdot \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \cdot \log(1 - \hat{y}^{(i)}) \Big]$$

*Where:*
* **$L$** is the final loss value for the batch
* **$m$** is the batch size
* **$y^{(i)}$** is the true label (0 or 1)
* **$\hat{y}^{(i)}$** is the predicted probability (between 0 and 1)

---

# 📈 Accuracy Function

#### **Sample-wise Error**
For each sample $i$, the absolute error is computed:
$$E_{i} = |\hat{y}^{(i)} - y^{(i)}|$$

#### **Compute Aggregated Error (Mean Absolute Error)**
The mean error across the batch of $m$ samples is calculated:
$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} E_{i}$$

#### **Compute Accuracy**
The final accuracy is defined as the complement of the mean absolute error:
$$\text{Accuracy} = 1 - \text{MAE}$$

---

## 📊 Results

---
