---
title: Model Distillation using Tensorflow, Pytorch and Google JAX
updated: 2023-01-16 10:38
---


<div class="subtitle">Knowledge distillation is a model compression technique whereby a small network (student) is taught by a larger trained neural network (teacher).</div>


<div class="divider"></div>

## I. What is model distillation?

Model distillation is a technique used to transfer knowledge from a larger, more complex model (the **"teacher" model**) to a smaller, simpler model (the **"student" model**) in order to improve the performance of the smaller model.

The basic idea is that the **teacher model** has already learned useful information from the data, and by distilling that knowledge into the **student model**, we can improve the **student model**'s performance without the need for more data or computational resources.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*EkkeLT74OgnGflvu" alt="Model Distillation">

<div class="divider"></div>

## II. Different Techniques to do model distillation

1. One of the most common is to use the **teacher model** to generate "soft" or "probabilistic" labels for the training data, and then use those labels to train the **student model**. This is done by having the **teacher model** generate a probability distribution over the possible outputs (e.g. class probabilities) for each input in the training data, and then using those probabilities as "pseudo-labels" for the **student model**'s training.

2. Another way to distill knowledge is by using the **teacher model**'s activations as guidance for the **student model**, which can be done by adding a term in the **student model**'s loss function which penalize the **student model**'s activations from deviating from the **teacher model**'s activations.

3. There are other ways to distill the knowledge too, such as using the **teacher model**'s output layers as regularizers for the **student model**'s output layers and etc.

<div class="divider"></div>

## III. Model Distillation using Tensorflow

Here's an example of using model distillation for binary classification using TensorFlow:

```python
import tensorflow as tf

# Define the teacher model
teacher_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Define the student model
student_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the teacher model
teacher_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate soft labels using teacher model
X_train, y_train = ... # load your dataset
soft_labels = teacher_model.predict(X_train, batch_size=32)

# Use soft labels to train the student model
student_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
student_model.fit(X_train, soft_labels, epochs=10, batch_size=32)
```


In this example, we define two models: a **teacher model** and a student model. The **teacher model** is a large and complex model with two hidden layers, and the **student model** is a smaller and simpler model with one hidden layer.

We first compile the **teacher model** and use it to generate soft labels for the training data by using the `predict()` method. These soft labels are then used to train the **student model** using the `fit()` method.

It's important to note that this is a simplified example, in practice, there are more details that need to be considered such as the temperature scaling for the soft labels and etc. Also, it's a good idea to fine-tune the student model on true labels as well.

<div class="divider"></div>
## IV. Model Distillation with Pytorch

Here's an example of using model distillation for ranking models using PyTorch:

```python
import torch
import torch.nn as nn

# Define the teacher model
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)
        
    def forward(self, user_id, item_id):
        user_embed = self.user_embedding(user_id)
        item_embed = self.item_embedding(item_id)
        concat = torch.cat([user_embed, item_embed], dim=-1)
        return self.fc(concat)
    
teacher_model = TeacherModel()

# Define the student model
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)
        
    def forward(self, user_id, item_id):
        user_embed = self.user_embedding(user_id)
        item_embed = self.item_embedding(item_id)
        concat = torch.cat([user_embed, item_embed], dim=-1)
        return self.fc(concat)

student_model = StudentModel()

# Generate soft labels using teacher model
X_train, y_train = ... # load your dataset
soft_labels = teacher_model(X_train).detach()

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(student_model.parameters())

# Use soft labels to train the student model
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = student_model(X_train)
    loss = loss_fn(output, soft_labels)
    loss.backward()
    optimizer.step()
```

In this example, we define two models: a **teacher model** and a **student model**. The **teacher model** is a large and complex model with two hidden layers, and the **student model** is a smaller and simpler model with one hidden layer.

We first compile the **teacher model** and use it to generate soft labels for the training data by using the `predict()` method. These soft labels are then used to train the **student model** using the `fit()` method.

It's important to note that this is a simplified example, in practice, there are more details that need to be considered such as the temperature scaling for the soft labels and etc. Also, it's a good idea to fine-tune the **student model** on true labels as well.

<div class="divider"></div>

## V. Model Distillation using Google Jax

Here's an example of using model distillation for ranking models using JAX:

```python
import jax
import jax.numpy as np
from jax import grad, jit
from jax.experimental import optimizers

# Define the teacher model
def teacher_model(params, user_id, item_id):
    user_embed, item_embed, w1, b1, w2, b2 = params
    user_embed = user_embed[user_id]
    item_embed = item_embed[item_id]
    concat = np.concatenate([user_embed, item_embed], axis=1)
    hidden = np.dot(concat, w1) + b1
    hidden = np.maximum(hidden, 0)
    return np.dot(hidden, w2) + b2

# Define the student model
def student_model(params, user_id, item_id):
    user_embed, item_embed, w, b = params
    user_embed = user_embed[user_id]
    item_embed = item_embed[item_id]
    concat = np.concatenate([user_embed, item_embed], axis=1)
    return np.dot(concat, w) + b

# Generate soft labels using teacher model
X_train, y_train = ... # load your dataset
soft_labels = teacher_model(teacher_params, X_train[:, 0], X_train[:, 1])

# Define loss function and optimizer
def loss_fn(params, x, y):
    return np.mean((student_model(params, x[:, 0], x[:, 1]) - y) ** 2)

# Use soft labels to train the student model
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)
student_params = opt_init(student_model.init_params)
for _ in range(num_epochs):
    grads = grad(loss_fn)(student_params, X_train, soft_labels)
    student_params = opt_update(grads, student_params)

```

In this example, we use the `optimizers.adam()` function to initialize the optimizer and set the step size. Then we initialize the **student model**'s parameters using the `init_params` function. Next, we run the training loop for a number of epochs.

In each iteration of the loop, we first calculate the gradients of the loss function with respect to the **student model**'s parameters using the `grad()` function. Then we update the **student model**'s parameters using the optimizer's `update()` function.

The **student model**'s parameters are updated at each iteration of the loop, causing the model to converge to a set of parameters that minimize the loss function, which in this case is the mean squared error between the **student model**'s predictions and the soft labels generated by the **teacher model**.
<div class="divider"></div>


## Appendix:

### Why use pseudo labels instead of the true labels

Using pseudo labels, instead of true labels, in model distillation is beneficial for a few reasons:

- The **teacher model** may have been trained on a larger and more diverse dataset than the **student model**, which means that it can generate more accurate and informative labels for the training data. Using these pseudo labels can help to improve the generalization of the **student model**.
- The **teacher model** may have learned to recognize patterns or features in the data that are not present in the true labels, but are still useful for the **student model** to learn. By using the pseudo labels, the **student model** can learn to extract these additional features, which can improve its performance.
- The **student model** may not have access to the true labels, or they may be expensive to obtain. In this case, using pseudo labels can be a more efficient and cost-effective way to train the **student model**.
- Using pseudo-labels can also increase the amount of data, which can have a positive effect on the performance of **student model**.

It's important to note that in some cases, the **student model** needs to be trained on the true labels as well, especially when the **student model** and **teacher model** are trained on different datasets. This is to prevent the **student model** from learning the noise in the pseudo labels.

### What is "soft" or "probabilistic" labels?

A "soft label" or "probabilistic label" is a probability distribution over the possible output classes, rather than a single discrete label. In other words, instead of assigning a single class label to each input, a soft label assigns a probability to each class, indicating the degree of confidence that the input belongs to that class.

For example, if we have a model that classifies images of animals into 10 different classes, a soft label for an image of a dog would be a 10-dimensional vector where the probability of the "dog" class is high, and the probabilities of the other classes are low.

Soft labels can be generated by a **teacher model** trained on the same dataset as the **student model**, or by any other model that is able to produce probability distributions over the output classes. These soft labels can then be used as "pseudo-labels" to train the **student model**, which can help it to learn more effectively and generalize better.

Soft labels can be useful in the context of model distillation and knowledge transfer because it helps to transfer more information from the **teacher model** to the **student model**. It allows the **student model** to learn from the **teacher model**'s confidence and uncertainty, which can help the **student model** to learn more effectively.

### Why the model is distilled (smaller) with more information passed in training process?

The **student model** is smaller than the **teacher model** because it has fewer parameters, which means that it requires less memory and computational resources to make predictions. This can be beneficial when deploying the model on resource-constrained devices or in situations where it is important to minimize the computational cost of making predictions.