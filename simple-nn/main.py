import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


def plt_mc(X, y):
    plt.figure(figsize=(6, 5))
    classes = np.unique(y)
    colors = ['r', 'g', 'b', 'y', 'pink', 'orange']

    for c in classes:
        points = X[y == c]
        plt.scatter(points[:, 0], points[:, 1],
                    color=colors[c], label=f'Class {c}')

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Training Data by Class")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # CoffeeRoasting data (Temperature, Duration) -> Delicious
    X = np.array([[100, 1], [200, 1], [300, 1], [400, 1], [250, 2], [300, 3]])
    Y = np.array([[0], [1], [1], [0], [1], [0]])

    print(X.shape, Y.shape)

    print(f"""
    pre normalization:
    Temperature Max, Min: {np.max(X[:, 0]):0.2f}, {np.min(X[:, 0]):0.2f}
    Duration Max, Min: {np.max(X[:, 1]):0.2f}, {np.min(X[:, 1]):0.2f}
    """)
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)  # learns mean, variance
    Xn = norm_l(X)
    print(f"""
    post normalization:
    Temperature Max, Min: {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}
    Duration Max, Min: {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}
    """)

    Xt = np.tile(Xn, (1000, 1))
    Yt = np.tile(Y, (1000, 1))
    print(Xt.shape, Yt.shape)

    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            tf.keras.Input(shape=(2,)),
            Dense(3, activation='relu', name='layer1'),
            Dense(1, activation='sigmoid', name='layer2')
        ]
    )

    # provides a description of the network
    model.summary()

    # (W1 parameters (input size)  + b1 parameters) * neurons
    L1_num_params = (2 + 1) * 3
    # (W2 parameters (input size)  + b2 parameters) * neurons
    L2_num_params = (3 + 1) * 1
    print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    )

    model.fit(Xt, Yt, epochs=10)

    W1, b1 = model.get_layer("layer1").get_weights()
    print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
    # W1(2, 3):
    #  [[ 6.4977016  6.033808   6.810511 ]
    #  [ 2.2868907 -3.389847  -0.9079014]]
    # b1(3,): [-4.430765  4.60956   6.558609]

    W2, b2 = model.get_layer("layer2").get_weights()
    print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)
    # W2(3, 1):
    #  [[-9.243866 ]
    #  [ 5.2992144]
    #  [ 3.6569176]]
    # b2(1,): [-3.809796]

    X_test = np.array([
        [150, 1],   # positive example
        [50, 1]])   # negative example
    X_testn = norm_l(X_test)
    predictions = model.predict(X_testn)
    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")
