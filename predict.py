# coding: utf-8

import sys

from tensorflow.python.keras.models import load_model
import numpy as np
import cv2

from binarize import binarize


def main():
    img_rows, img_cols = 400, 400
    img_channels = 1

    model = load_model('models/200.h5')
    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    binary = cv2.resize(binarize(img), (img_rows, img_cols))

    img2 = np.zeros((img_rows, img_cols, img_channels))
    img2[:, :, 0] = binary
    img2 = np.expand_dims(img2, axis=0)

    pred = model.predict(img2)
    pred = pred.flatten()
    predicted_class_indices = np.argmax(pred)

    labels = ['FBMessanger', 'Instagram', 'Invalid', 'LINE', 'Others', 'Pairs', 'Twitter']
    predictions = dict(zip(labels, pred))
    print("predictions:", predictions)
    print("service:", labels[predicted_class_indices])


if __name__ == '__main__':
    main()
