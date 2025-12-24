import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# make 6-class dataset for classification
X_train, y_train = make_blobs(
    n_samples=200, cluster_std=0.8, random_state=20,
    centers=np.array([[-5, 2], [-2, -2], [1, 2],
                        [4, -2], [7, 2], [10, -2]]),
)

plt.figure(figsize=(6, 5))
classes = np.unique(y_train)
colors = ['r', 'g', 'b', 'y', 'pink', 'orange']

for c in classes:
    points = X_train[y_train == c]
    plt.scatter(points[:, 0], points[:, 1],
                color=colors[c], label=f'Class {c}')

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Training Data by Class")
plt.legend()
plt.grid(True)
plt.show()

print(f"unique classes {np.unique(y_train)}")
print(f"10 class representation {y_train[:10]}")
print(
    f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        Dense(3, activation='relu',   name="L1"),
        Dense(6, activation='linear', name="L2")
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

model.fit(X_train, y_train, epochs=200)

l1 = model.get_layer("L1")
W1, b1 = l1.get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)

l2 = model.get_layer("L2")
W2, b2 = l2.get_weights()
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

prediction = model.predict(X_train)

for i in range(5):
    print(f"""
    X_train:
    {X_train[i]}
    prediction:
    {prediction[i]}
    category:
    {np.argmax(prediction[i])}
    """)
