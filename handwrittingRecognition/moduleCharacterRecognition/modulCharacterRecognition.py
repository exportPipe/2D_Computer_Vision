import tensorflow as tf
import numpy as np
import pytesseract


def predictCharacterWithCNN(imagesIn):
    model = tf.keras.models.load_model("cnna5")
    model.trainable = False

    pred = model.predict(imagesIn)

    return toString(np.argmax(pred, axis=1))


def predictCharacterWithOCR(imagesIn):
    string = ''
    for image in imagesIn:
        string.append(pytesseract.image_to_string(image))

    return string


def toString(prediction):
    pass


def getString(imagesIn):
    return predictCharacterWithCNN(imagesIn)
