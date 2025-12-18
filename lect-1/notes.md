# Lecture 1 — Introduction to Deep Learning

**Course:** MIT 6.S191 (2025)  
**Date:** December 15, 2025  
**Instructor:** Alexander Amini

**Video:**
[Link](https://www.youtube.com/watch?v=alfdI7S6wCY&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1)
**Slides:**
[Link](https://introtodeeplearning.com/2025/slides/6S191_MIT_DeepLearning_L1.pdf)

---

## 📌 Executive Summary

This lecture introduces the **foundations of deep learning**, tracing the progression from biologically inspired neuron models to modern **deep neural networks**.

Core ideas include:
- Why deep learning works and why it became practical only recently
- The perceptron and dense layers
- Forward propagation, activation functions, and loss functions
- Training neural networks using **gradient descent** and **backpropagation**
- Practical considerations such as optimizers, stochastic training, overfitting, and regularization

---

## 🧠 Core Concepts & Terminology

- **Artificial Intelligence (AI):** Techniques that enable machines to mimic intelligent behavior.
- **Machine Learning (ML):** A subset of AI where models learn patterns from data instead of explicit rules.
- **Deep Learning (DL):** A subset of ML that uses multi-layer neural networks to learn hierarchical representations.
- **Perceptron:** A single artificial neuron that computes a weighted sum followed by a non-linear activation.
- **Dense (Fully Connected) Layer:** A layer where each input neuron connects to every output neuron.
- **Forward Propagation:** Computing outputs by passing inputs through network layers.
- **Activation Function:** A non-linear function applied to neuron outputs (e.g., ReLU, Sigmoid, Tanh).
- **Loss Function:** Quantifies the error between predictions and ground truth.
- **Gradient Descent:** An optimization method that minimizes loss by updating weights in the direction of negative gradients.
- **Backpropagation:** Efficient computation of gradients using the chain rule.
- **Learning Rate:** Controls the step size of weight updates.
- **Overfitting:** When a model memorizes training data and fails to generalize.
- **Regularization:** Techniques that improve generalization by discouraging overly complex models.
- **Stochastic Gradient Descent (SGD):** Gradient descent using mini-batches instead of the full dataset.

---

## 📝 Detailed Notes

---

## 1. Why Deep Learning? Why Now?

Traditional machine learning relied heavily on **hand-engineered features**, which were:
- Time-consuming
- Brittle
- Difficult to scale across domains

Deep learning replaces feature engineering with **feature learning**, allowing models to automatically learn useful representations from raw data.

Deep learning has existed for decades — so why did it take off recently?

**Three key enablers:**

1. **Data:** Massive datasets enabled by the internet and digitization
2. **Computation:** GPUs and TPUs enabling large-scale parallel computation
3. **Software:** Open-source frameworks (PyTorch, TensorFlow, JAX) lowering the barrier to entry

---

## 2. The Perceptron: A Single Neuron

### Forward Propagation

![The Perceptron](assets/image.png)

A perceptron computes:
- A weighted sum of inputs
- Adds a bias term
- Applies a non-linear activation function

Given inputs $x_1, \dots, x_m$ and weights $w_1, \dots, w_m$:

$$\hat{y} = g\left(w_0 + \sum_{i=1}^{m} w_i x_i\right)$$

Vectorized form:

$$\hat{y} = g\left(w_0 + \mathbf{X}^T \mathbf{W}\right)$$

where

$$\mathbf{X} = [x_1, \dots, x_m], \quad \mathbf{W} = [w_1, \dots, w_m]$$

The **bias** shifts the activation function, improving flexibility.

---

### Non-Linear Activation Functions

![Common Activation Function](assets/image-1.png)

Activation functions introduce non-linearity, allowing networks to model complex patterns.

Common examples:

**Sigmoid:**
$$g(x) = \frac{1}{1 + e^{-x}}$$

**Tanh:**
$$g(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**ReLU:**
$$g(x) = \max(0, x)$$

#### Framework Implementations

**PyTorch:**
```python
torch.sigmoid(x)
torch.tanh(x)
torch.relu(x)
```

**TensorFlow:**
```python
tf.sigmoid(x)
tf.tanh(x)
tf.nn.relu(x)
```

---

## 3. Multi-Output Perceptron (Dense Layer)

![Dense Layer](assets/image-2.png)

- Multiple perceptrons operating on the same input form a **dense layer**
- Each neuron has its own weights and bias
- Without an activation function, a dense layer is purely **linear**

### Dense Layer from Scratch

#### PyTorch

```python
class MyDenseLayer(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(input_dims, output_dims))
        self.b = torch.nn.Parameter(torch.randn(1, output_dims))

    def forward(self, inputs):
        z = torch.matmul(inputs, self.w) + self.b
        return torch.relu(z)
```

**Built-in:** `nn.Linear(in_features, out_features)`

#### TensorFlow

```python
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.w = self.add_weight(shape=(input_dims, output_dims))
        self.b = self.add_weight(shape=(output_dims,))

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        return tf.nn.relu(z)
```

**Built-in:** `tf.keras.layers.Dense(units, activation)`

---

## 4. Single Hidden Layer Neural Network

![Single Layer NN](assets/image-3.png)

- Introduces a **hidden layer** between input and output
- Hidden layers learn intermediate representations
- Called "hidden" because their activations are not directly observed

**PyTorch:**
```python
nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)
```

**TensorFlow:**
```python
tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, activation='relu'),
    tf.keras.layers.Dense(output_size)
])
```

---

## 5. Deep Neural Networks

![Deep Neural Network](assets/image-4.png)

- A **deep network** stacks multiple hidden layers
- Each layer learns increasingly abstract features

---

## 6. Loss Functions

![Loss Illustration](assets/image-5.png)

Loss quantifies how incorrect a model's predictions are.

### Binary Cross-Entropy Loss

Used for probabilistic binary classification.

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]$$

**PyTorch:**
```python
torch.nn.functional.cross_entropy(predicted, y)
```

**TensorFlow:**
```python
tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(y, predicted)
)
```

### Mean Squared Error (MSE)

Used for continuous regression targets.

$$L = \frac{1}{2N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**PyTorch:**
```python
torch.nn.MSELoss(predicted, y)
```

**TensorFlow:**
```python
tf.keras.losses.MeanSquaredError(predicted, y)
```

---

## 7. Training Neural Networks

![Training Objective](assets/image-6.png)

**Goal:**

$$W^* = \arg\min_W \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f(x^{(i)}; W), y^{(i)})$$

### Optimization via Gradient Descent

![Loss Landscape](assets/image-7.png)

- Loss is a function of network weights
- Gradients indicate the direction of steepest ascent
- We move **opposite** the gradient

![Gradient Descent](assets/image-8.png)

### Training Loop

**PyTorch:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**TensorFlow:**
```python
optimizer = tf.keras.optimizers.SGD(0.01)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(x_train)
        loss = compute_loss(y_train, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

---

## 8. Backpropagation

![Backpropagation](assets/image-9.png)

Gradients are computed using the **chain rule**, propagating error backward layer by layer.

$$\frac{\partial J}{\partial w_1} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

---

## 9. Optimizers

| Optimizer | PyTorch | TensorFlow |
|-----------|---------|------------|
| SGD | `torch.optim.SGD(...)` | `tf.keras.optimizers.SGD(...)` |
| Adam | `torch.optim.Adam(...)` | `tf.keras.optimizers.Adam(...)` |
| RMSprop | `torch.optim.RMSprop(...)` | `tf.keras.optimizers.RMSprop(...)` |
| Adagrad | `torch.optim.Adagrad(...)` | `tf.keras.optimizers.Adagrad(...)` |
| Adadelta | `torch.optim.Adadelta(...)` | `tf.keras.optimizers.Adadelta()` |

---

## 10. Stochastic Gradient Descent

![SGD](assets/image-11.png)

- Gradients computed on **mini-batches**
- Faster, memory-efficient, GPU-friendly
- Adds beneficial noise that improves generalization

---

## 11. Overfitting & Regularization

![Overfitting](assets/image-12.png)

Overfitting occurs when training performance improves but test performance degrades.

### Regularization Techniques

#### Dropout

![Dropout](assets/image-13.png)

- Randomly zeroes activations during training
- Common dropout rate: 50%
- Forces robust feature learning

#### Early Stopping

![Early Stopping](assets/image-14.png)

- Stop training when validation loss increases
- Prevents memorization


## 📚 References

- MIT 6.S191: [http://introtodeeplearning.com/](http://introtodeeplearning.com/)