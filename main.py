from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


def load_data() -> Bunch:
    return datasets.load_digits()


def prepare_data(data: Bunch, shuffle=False, test_size=0.15) -> Tuple[np.array, np.array, np.array, np.array]:

    images = data.images.reshape(len(data.images), -1)

    return train_test_split(images, data.target, test_size=test_size, shuffle=shuffle)


def load_custom_image(image_path: str) -> np.array:
    image = Image.open(image_path).convert('L')
    image = Image.eval(image, lambda pixel: 255 - pixel)
    image.thumbnail((8, 8), Image.Resampling.LANCZOS)

    new_image = Image.new('L', (8, 8), color=0)
    new_image.paste(image, ((8 - image.width) // 2, (8 - image.height) // 2))

    image_np = np.array(new_image) / 255.0

    scaled_image_np = image_np * 150
    rounded_image_np = np.round(scaled_image_np)

    return rounded_image_np


if __name__ == '__main__':
    digits = load_data()

    X_train, X_test, y_train, y_test = prepare_data(digits, test_size=0.12)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='sigmoid'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # training
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1)

    # test
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)

    # result
    print(metrics.classification_report(y_test, predictions, zero_division=0))

    # custom images classification
    custom_images_X = []
    custom_predictions_y = []

    for i in range(10):
        custom_predictions_y.append(i)
        custom_image = (load_custom_image(f'images/{i}.png'))
        custom_images_X.append(custom_image)

    for i in range(10):
        custom_predictions_y.append(i)
        custom_image = (load_custom_image(f'images2/{i}.png'))
        custom_images_X.append(custom_image)

    custom_images = np.array(custom_images_X)
    custom_flat_images = custom_images.reshape(len(custom_images), -1)

    custom_image_predictions = model.predict(custom_flat_images)
    custom_image_predictions = np.argmax(custom_image_predictions, axis=1)
    print(metrics.classification_report(custom_predictions_y, custom_image_predictions, zero_division=0))

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()

    disp2 = metrics.ConfusionMatrixDisplay.from_predictions(custom_predictions_y, custom_image_predictions)
    disp2.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp2.confusion_matrix}")
    plt.show()
