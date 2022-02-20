from image_processor import *
import digit_recognizer
import cv2 as cv
import numpy as np

#constants
SOURCE = "imgs/examples/1.jpg"
DESTINATION = "upload/skuska.jpg"
USED_METHOD = "CNN"

class GridDetector():
    def __init__(self, strategy):
        self._strategy = strategy
        self.dimensions= strategy.IMG_DIMENSIONS
        self.thread = strategy.NUMBER_THREAD
    
    @property
    def strategy(self):
        return self._strategy

    def get_grid(self, numbers: list) -> np.ndarray:
        """
        Evaluates images based on used algorithm and returns numpy.ndarray of sudoku grid
        """
        self._strategy.train()
        grid = np.zeros((9,9), dtype=np.int32)

        for index, number in enumerate(numbers):
            x, y = index % 9, int(index / 9)
            grid[y,x] = 0 if number is None else self._strategy.classify(number)
        return grid    


def main():
    """Run this first."""

    if USED_METHOD is "CNN":
        method = digit_recognizer.ConvolutionalNeuralNetwork
    elif USED_METHOD is "KNN":
        method = digit_recognizer.KNearestNeighbor
        
    context = GridDetector(method())

    sudoku_extractor = SudokuExtraction(SOURCE)
    if sudoku_extractor.find_sudoku_grid():
            sudoku_extractor.zoom_sudoku()
            numbers = sudoku_extractor.get_numbers()
            x = NumbersProcessing(numbers, method.IMG_DIMENSIONS)
            numbers = x.preprocess_numbers()
    else:
        numbers = []

    grid = context.get_grid(numbers)
    print(grid)

if __name__ == "__main__":
    main()