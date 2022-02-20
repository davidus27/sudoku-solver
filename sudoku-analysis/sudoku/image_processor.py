import cv2 as cv
import numpy as np

#STEP BY STEP:
#
#   1. Creating self.image object:
#               *Open a file
#               *Convert to grayscale
#               *Resize
#   2. Denoise
#   3. Adaptive thresholding
#   4. Eliminate grid lines
#   5. Finding numbers
#   6. Create numpy object

class SudokuExtraction(object):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.image = cv.imread(image_path)        
    
    def find_sudoku_grid(self):
        gray = cv.cvtColor(self.image,cv.COLOR_BGR2GRAY)
        processed_temp_image = cv.GaussianBlur(gray, (7,7), 0)
        processed_temp_image = cv.adaptiveThreshold(processed_temp_image, 
                                                        255,
                                                        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv.THRESH_BINARY_INV,
                                                        11,
                                                        2)
        countours, _ = cv.findContours(processed_temp_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        if not countours:
            return False # no contour found
            
        sudoku_grid = sorted(countours, key=cv.contourArea, reverse=True)[0]
        epsilon = 0.1*cv.arcLength(sudoku_grid,True)
        
        self.grid_edges = cv.approxPolyDP(sudoku_grid, epsilon,True)
        self.sudoku_grid = cv.drawContours(self.image, [self.grid_edges], -1, (255,255,255), 3)
        return True

    def rotate_image(self, angle):
        image_center = tuple(np.array(self.image.shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
        return cv.warpAffine(self.image, rot_mat, self.image.shape[1::-1], flags=cv.INTER_LINEAR)

    @property
    def longest_side(self):
        return max(
                    [
                        cv.norm(self.grid_edges[0][0], self.grid_edges[1][0]), # top left vs top right
                        cv.norm(self.grid_edges[1][0], self.grid_edges[2][0]), # top right vs bottom right
                        cv.norm(self.grid_edges[2][0], self.grid_edges[3][0]), # bottom right vs bottom left
                        cv.norm(self.grid_edges[3][0], self.grid_edges[0][0])  # bottom left vs top left
                    ]
                )
                
    def zoom_sudoku(self):
        src = np.array([self.grid_edges[1][0],
                            self.grid_edges[0][0],
                            self.grid_edges[3][0],
                            self.grid_edges[2][0]], dtype=np.float32) 

        dst = np.array([[0, 0],
                            [self.longest_side - 1, 0], 
                            [self.longest_side - 1, self.longest_side - 1], 
                            [0, self.longest_side - 1]], dtype=np.float32)

        tranformed_perspective = cv.getPerspectiveTransform(src, dst)
        transformed_image = cv.warpPerspective(self.image,
                                                tranformed_perspective,
                                                (int(self.longest_side), int(self.longest_side)))
        self.sudoku_grid = self.rotate_image(90) # TODO: check if it works

    def get_numbers(self):
        processed_image = cv.cvtColor(self.sudoku_grid,cv.COLOR_BGR2GRAY)
        cv.bitwise_not(processed_image, processed_image)
        for i in range(9):
            for j in range(9):
                width = int(processed_image.shape[0]/9)
                height = int(processed_image.shape[1]/9)
                yield processed_image[i*width:i*width + width,j*height:j*height + height]



class NumbersProcessing:
    # Crop the image
    UP = 0.15
    LEFT = 0.2
    # Thresholds
    NUMBER_THREASHOLD = 150
    MIN_NUMBER_SIZE = 75

    def __init__(self, numbers, resolution):
        self.numbers = numbers
        self.resolution = resolution

    def _find_center(self, contours):
        M = cv.moments(sorted(contours, key=cv.contourArea, reverse=True)[0])
        return int(M['m10']/M['m00']), int(M['m01']/M['m00']) # cx, cy
    
    def _center_number(self, number):
        contours, _ = cv.findContours(number, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cx, cy = self._find_center(contours)
        number = cv.drawContours(np.zeros((cy*2, cx*2)), contours, -1, (255,255,255))
        number = cv.fillPoly(number,pts=contours, color=(255,255,255))
        number = cv.resize(number, (self.resolution,self.resolution))
        return cv.erode(number, (3,3))

    def preprocess_numbers(self):
        for num in self.numbers:
            num = num[int(num.shape[0]*self.UP):num.shape[0],int(num.shape[1]*self.LEFT):num.shape[1]]
            num = cv.resize(num, (60,60))
            _,num = cv.threshold(num,self.NUMBER_THREASHOLD,255,cv.THRESH_BINARY)
            
            if cv.countNonZero(num) > self.MIN_NUMBER_SIZE:
                yield self._center_number(number=num)
            else:
                yield None
