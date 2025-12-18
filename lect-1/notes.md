# Lecture 1 — Introduction to Deep Learning

**MIT 6.S191 (2025)**

**Date:** December 15, 2025

**Instructor:** Alexander Amini

**Video:**
[Link](https://www.youtube.com/watch?v=alfdI7S6wCY&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1)

**Slides:**
[Link](https://introtodeeplearning.com/2025/slides/6S191_MIT_DeepLearning_L1.pdf)

---

## 1. Overview

This lecture introduces the **foundations of Deep Learning**, beginning with simple neuron models and building up to modern deep neural networks. It focuses on both **intuition** and **mechanics**: how networks are constructed, how they learn, and why they generalize (or fail to).

Key topics include:

* The perceptron and dense layers
* Forward propagation and activation functions
* Loss functions and optimization
* Backpropagation and gradient descent
* Overfitting and regularization

---

## 2. Why Deep Learning? Why Now?

Traditional machine learning systems relied on **hand-engineered features**, which were:

* Time-consuming
* Brittle to changes in data
* Hard to scale across domains

Deep learning replaces manual feature design with **automatic feature learning**.

### Key Enablers

1. **Data** — Large-scale datasets from the internet and digital platforms
2. **Computation** — GPUs and TPUs enabling massive parallelism
3. **Software** — Frameworks like PyTorch, TensorFlow, and JAX

---

## 3. The Perceptron (Single Neuron)

The **perceptron** is the fundamental computational unit of a neural network.

![Figure 1: Single perceptron model](assets/image.png)

**Figure 1 intuition:**
Each input contributes proportionally to the output via its weight. The bias shifts the activation, and the non-linearity enables expressive modeling.

### Forward Propagation

$$
\hat{y} = g\left(w_0 + \sum_{i=1}^{m} w_i x_i \right)
$$

Vectorized form:

$$
\hat{y} = g\left(w_0 + \mathbf{X}^\top \mathbf{W}\right)
$$

Where:

* $$ \mathbf{X} = [x_1, \dots, x_m] $$
* $$ \mathbf{W} = [w_1, \dots, w_m] $$
* $$ g(\cdot) $$ is an activation function

---

## 4. Activation Functions

Activation functions introduce **non-linearity**, allowing neural networks to model complex relationships.

![Figure 2: Common activation functions and derivatives](assets/image-1.png)

**Figure 2 intuition:**
Without non-linear activations, stacked layers collapse into a single linear transformation.

### Common Functions

* **Sigmoid**

$$
g(x) = \frac{1}{1 + e^{-x}}
$$

* **Tanh**

$$
g(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

* **ReLU**

$$
g(x) = \max(0, x)
$$

---

## 5. Multi-Output Perceptron (Dense Layer)

Multiple perceptrons can operate in parallel to produce **multiple outputs** from the same input.

![Figure 3: Multi-output perceptron (dense layer)](assets/image-2.png)

**Figure 3 intuition:**
Each neuron has its own weights, allowing the layer to learn multiple features simultaneously.

Key properties:

* Fully connected (dense)
* Linear if no activation is applied
* Forms the core building block of neural networks

---

## 6. Single Hidden Layer Neural Network

Adding a **hidden layer** enables the network to learn intermediate representations.

![Figure 4: Single hidden layer neural network](assets/image-3.png)

**Figure 4 intuition:**
Hidden layers transform raw inputs into increasingly abstract features.

---

## 7. Deep Neural Networks

Deep neural networks stack multiple hidden layers hierarchically.

![Figure 5: Deep neural network architecture](assets/image-4.png)

**Figure 5 intuition:**
Lower layers learn simple features; deeper layers combine them into high-level concepts.

---

## 8. Loss Functions

To learn, a network must quantify **how wrong its predictions are**.

![Figure 6: Empirical loss over a dataset](assets/image-5.png)

**Figure 6 intuition:**
Training minimizes the *average* loss across all data points.

### Binary Cross-Entropy Loss

$$
L = - \frac{1}{N} \sum_{i=1}^{N}
\left[
y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)
\right]
$$

Used for probabilistic classification.

---

### Mean Squared Error (MSE)

$$
L = \frac{1}{2N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Used for regression tasks.

---

## 9. Training Neural Networks

Training seeks parameters that minimize the loss function.

$$
W^* = \arg\min_W \frac{1}{n} \sum_{i=1}^{n}
\mathcal{L}(f(x^{(i)}; W), y^{(i)})
$$

![Figure 7: Training objective](assets/image-6.png)

---

## 10. Gradient Descent

The loss is a function of the weights and can be visualized as a surface.

![Figure 8: Loss landscape](assets/image-7.png)

**Figure 8 intuition:**
Optimization is a search for the lowest point in this landscape.

### Gradient Descent Steps

![Figure 9: Gradient descent updates](assets/image-8.png)

**Figure 9 intuition:**
The gradient points uphill; we step in the opposite direction.

---

## 11. Backpropagation

Gradients are computed efficiently using the **chain rule**.

![Figure 10: Backpropagation through the network](assets/image-9.png)

$$
\frac{\partial J}{\partial w_1}
===============================

\frac{\partial J}{\partial \hat{y}}
\cdot
\frac{\partial \hat{y}}{\partial z_1}
\cdot
\frac{\partial z_1}{\partial w_1}
$$

**Figure 10 intuition:**
Errors flow backward, distributing responsibility across layers.

---

## 12. Stochastic Gradient Descent

Computing gradients over the full dataset is expensive.

![Figure 11: Full-batch vs stochastic gradient descent](assets/image-10.png)

Solution: **mini-batch SGD**

![Figure 12: Mini-batch gradient descent](assets/image-11.png)

**Figure 12 intuition:**
Mini-batches reduce computation and introduce noise that helps escape poor minima.

---

## 13. Overfitting

A model may perform well on training data but fail to generalize.

![Figure 13: Overfitting vs generalization](assets/image-12.png)

**Figure 13 intuition:**
Generalization matters more than training accuracy.

---

## 14. Regularization Techniques

### Dropout

![Figure 14: Dropout during training](assets/image-13.png)

**Figure 14 intuition:**
Randomly disabling neurons forces redundancy and robustness.

---

### Early Stopping

![Figure 15: Early stopping based on validation loss](assets/image-14.png)

**Figure 15 intuition:**
Stop training when validation performance degrades, even if training loss continues to decrease.

---

## 15. References

* MIT 6.S191 — Introduction to Deep Learning
  [http://introtodeeplearning.com/](http://introtodeeplearning.com/)
