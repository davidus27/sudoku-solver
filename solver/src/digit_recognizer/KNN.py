from digit_recognizer.digit_recognizer import DigitRecognizer
import numpy as np
import cv2

"""
Process:
    * train_imgs it on 20x20
    * For all numbers:
        - resize (20x20)
        - threashold
        - draw contour (EXTERNAL) on blank picture (20x20) with 3pix conversion
    * Evaluate all numbers
"""


#decorators
def timer(fun):
    from time import time
    def wrapper():
        start = time()
        fun()
        print(time() - start)
    return wrapper

def ntimer(n):
    from time import time
    def wrapper(fun):
        start = time()
        for i in range(n): 
            fun()
        print(time() - start)
    return wrapper

IMAGE_PATH = "imgs/learning_data/digits.png"

class KNearestNeighbor(DigitRecognizer):
    IMG_DIMENSIONS=20
    PIXEL=400
    K=5
    NUMBER_THREAD=100
    def __init__(self):
        self.images = self._split_image(IMAGE_PATH, 100, 50)
        self.train_imgs = None
        self.test_imgs = None
        self.train_labels = None
        self.test_labels = None
        self.model = cv2.ml.KNearest_create()

    def classify(self, image: np.ndarray) -> int:
        if self.train_imgs is None or self.train_labels is None:
            self.train()
        image = image.reshape(-1, self.PIXEL).astype(np.float32)
        return self.model.findNearest(image, k = self.K)[1]

    def train(self) -> None:
        if not self.images.any():
            return
        #create training data
        self.train_imgs = self.images[:,:50].reshape(-1,self.PIXEL).astype(np.float32)
        self.test_imgs = self.images[:,50:100].reshape(-1,self.PIXEL).astype(np.float32)
        
        #make training labels
        k = np.arange(10)
        self.train_labels = np.repeat(k,250)[:,np.newaxis]
        self.test_labels = self.train_labels.copy()

        #create knn object and train_imgs the model
        self.model.train(self.train_imgs, cv2.ml.ROW_SAMPLE, self.train_labels)

    def is_trained(self) ->bool:
        return self.train_imgs and self.test_imgs and self.train_labels and self.test_labels

    def check_precision(self, labels, results):
        if results is None:
            return None 
        correct = np.count_nonzero(results == labels)
        return correct/results.size

    def save_data(self, name):
        np.savez(name,train=self.train_imgs, train_labels=self.train_labels)
        
    def load_model(self, name):
        # Now load the data
        with np.load(name) as data:
            self.train_imgs = data['train']
            self.train_labels = data['train_labels']

    def _split_image(self, image_path, width, height): 
        # Split image for training the model
        image = cv2.imread(image_path, 0)
        if image is None:
            return None
        # Now we split the image to 5000 cells, each 20x20 size
        cells = [np.hsplit(row,100) for row in np.vsplit(image,50)]

        # Make it into a Numpy array. It size will be (50,100,20,20)
        return np.array(cells)