import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from extra_keras_datasets import emnist


def trainCnn():
    (input_train, target_train), (input_test, target_test) = emnist.load_data(type='letters')

    in_sort_train = np.where(target_train)
    in_sort_test = np.where(target_test)
    x_train = input_train[in_sort_train]
    y_train = target_train[in_sort_train]

    x_test = input_test[in_sort_test]
    y_test = target_test[in_sort_test]

    # normalize data set
    valIdx = int(len(y_train))
    trainIdx = int(valIdx * 0.8)

    X_train = x_train[0:trainIdx] / 255  # divide by 255 so that they are in range 0 to 1
    Y_train = keras.utils.to_categorical(y_train[0:trainIdx])  # one-hot encoding

    X_val = x_train[trainIdx:valIdx] / 255
    Y_val = keras.utils.to_categorical(y_train[trainIdx:valIdx])

    X_test = x_test / 255
    Y_test = keras.utils.to_categorical(y_test)

    # plot first few elements of modified emnist (just hex)
    plt.figure(figsize=(12, 12))
    for i in range(0, 7):
        plt.subplot(1, 7, (i + 1))
        plt.imshow((X_test[i + 1000]), cmap="gray")
        plt.title('true label: ' + str(np.argmax(Y_test, axis=1)[i + 1000]))

    # shape data set
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_val = np.reshape(X_val, (X_val.shape[0], 28, 28, 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1), activation='relu'),
        tf.keras.layers.AveragePooling2D(2, 2),
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation='elu'),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(27, activation='softmax')
    ])

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam',
                  metrics=['accuracy'])

    # fit model with training set
    history = model.fit(X_train, Y_train,
                        batch_size=128,
                        epochs=10,
                        verbose=2,
                        validation_data=(X_val, Y_val),
                        shuffle=True
                        )

    model.save("cnnModel")
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], linestyle='-.')
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy cnn')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='lower right')
    plt.subplot(2, 2, (2))
    plt.plot(history.history['loss'], linestyle='-.')
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss cnn')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    trainCnn()