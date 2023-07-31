from digit_recognizer.digit_recognizer import DigitRecognizer
import numpy as np
import os

try:
    CAN_TRAIN = True
    from tensorflow.keras import utils
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
except ModuleNotFoundError:
    CAN_TRAIN = False

CHECKPOINT_PATH="trained_model/cp.ckpt"
EPOCHS_AMOUNT=5
CLASSES_AMOUNT=10
BATCH_SIZE=200


class ConvolutionalNeuralNetwork(DigitRecognizer):
    IMG_DIMENSIONS=28
    NUMBER_THREAD=100

    def __init__(self):
        self.model = None
        self.image_train = None 
        self.labels_train = None
        self.image_test = None
        self.labels_test = None
        self.input_shape = (self.IMG_DIMENSIONS, self.IMG_DIMENSIONS, 1)

    def _create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(CLASSES_AMOUNT, activation='softmax'))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return self.model

    def _prepare_data(self):
        (self.image_train, self.labels_train) , (self.image_test, self.labels_test) = mnist.load_data()

        self.image_train = self.image_train.reshape(self.image_train.shape[0],
                                                        self.IMG_DIMENSIONS,
                                                        self.IMG_DIMENSIONS,
                                                        1)
        self.image_train = self.image_train.astype(np.float32) / 255
        
        self.image_test = self.image_test.reshape(self.image_test.shape[0],
                                                    self.IMG_DIMENSIONS,
                                                    self.IMG_DIMENSIONS,
                                                    1)
        self.image_test = self.image_test.astype(np.float32) / 255

        self.labels_train = utils.to_categorical(self.labels_train, 
                                                    num_classes=CLASSES_AMOUNT)
        self.labels_test = utils.to_categorical(self.labels_test, 
                                                    num_classes=CLASSES_AMOUNT)

    def _prepare_model(self):
        checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)

        cp_callback = ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                        save_weights_only=True,
                                        verbose=1)
        self.model.fit(self.image_train,
                    self.labels_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS_AMOUNT,
                    verbose=1,
                    validation_data=(self.image_test, self.labels_test),
                    callbacks=[cp_callback])

    def _trained_model_found(self):
        """ Checkpoint directory is not empty """
        return os.path.isfile(CHECKPOINT_PATH) 

    def train(self):
        if self._trained_model_found():
            self.load_model()
            return
        if CAN_TRAIN:
            self._create_model()
            self._prepare_data()
            self._prepare_model()
        else:
            print("You need to install Tensorflow")  

    def is_trained(self):
        return bool(self.model)

    def check_precision(self):
        return new_model.evaluate(image_test,  labels_test, verbose=2)[1]

    def classify(self, image : np.ndarray) -> int:
        return self.model.predict(image.reshape(1, self.IMG_DIMENSIONS, self.IMG_DIMENSIONS, 1)).argmax()
         
    def load_model(self, filename=CHECKPOINT_PATH):
        self.model = self._create_model()
        self.model.load_weights(filename)
        return True