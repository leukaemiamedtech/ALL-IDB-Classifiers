############################################################################################
#
# Project:       Peter Moss Leukemia Research Foundation
# Repository:    ALL-IDB Classifiers
# Project:       Tensorflow 2.0 ALL Papers
#
# Author:        Adam Milton-Barker (adammiltonbarker@leukemiaresearchfoundation.ai)
# Contributors:
#
# Title:         Paper 1 Data Augmentation Helper Class
# Description:   Data augmentation helper class for the Paper 1 Evaluation.
# License:       MIT License
# Last Modified: 2019-12-30
#
############################################################################################

import cv2
import random

import numpy as np

from numpy.random import seed
from scipy import ndimage
from skimage import transform as tm

from Classes.Helpers import Helpers


class Augmentation():
    """ Data Augmentation Class
    Data augmentation helper class for the Paper 1 Evaluation.
    """

    def __init__(self, model, optimizer):
        """ Initializes the Data Augmentation class. """

        self.Helpers = Helpers("Augmentation", False)
        self.model_type = model
        self.optimizer = optimizer
        self.seed = self.Helpers.confs[self.model_type]["data"]["seed_" + self.optimizer + "_augmentation"]
        
        seed(self.seed)
        random.seed(self.seed)
        
        self.Helpers.logger.info("Data augmentation class initialization complete.")

    def grayscale(self, data):
        """ Creates a grayscale copy. """

        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        return np.dstack([gray, gray, gray]).astype(np.float32)/255.

    def equalize_hist(self, data):
        """ Creates a histogram equalized copy. 
        
        Credit: Amita Kapoor & Taru Jain
        Exploring novel convolutional network architecture to build a classification 
        system for better assistance in diagonosing Acute Lymphoblastic Leukemia in 
        blood cells.


        https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Keras/QuantisedCode/QuantisedCode.ipynb
        """

        img_to_yuv = cv2.cvtColor(data, cv2.COLOR_BGR2YUV)
        img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
        hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
        return hist_equalization_result.astype(np.float32)/255.

    def reflection(self, data):
        """ Creates a reflected copy. """

        return cv2.flip(data, 0).astype(np.float32)/255., cv2.flip(data, 1).astype(np.float32)/255.

    def gaussian(self, data):
        """ Creates a gaussian blurred copy. """

        return ndimage.gaussian_filter(data, sigma=5.11).astype(np.float32)/255.

    def translate(self, data):
        """ Creates transformed copy. """

        cols, rows, chs = data.shape

        return cv2.warpAffine(data, np.float32([[1, 0, 84], [0, 1, 56]]), (rows, cols),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(144, 159, 162)).astype(np.float32)/255.

    def rotation(self, data, label, tdata, tlabels):
        """ Creates rotated copies. """

        cols, rows, chs = data.shape

        for i in range(0, self.Helpers.confs[self.model_type]["data"]["rotations_augmentation"]):
            rand_deg = random.randint(-180, 180)
            matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rand_deg, 0.70)
            rotated = cv2.warpAffine(data, matrix, (rows, cols), borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(144, 159, 162))

            rotated = rotated.astype(np.float32)/255.

            tdata.append(rotated)
            tlabels.append(label)
            
        return tdata, tlabels

    def shear(self, data):
        """ Creates a histogram equalized copy. 
        
        Credit: Amita Kapoor & Taru Jain
        Exploring novel convolutional network architecture to build a classification 
        system for better assistance in diagonosing Acute Lymphoblastic Leukemia in 
        blood cells.


        https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Keras/QuantisedCode/QuantisedCode.ipynb
        """

        at = tm.AffineTransform(shear=0.5)
        return tm.warp(data, inverse_map=at)

    def write_image(self, floc, data):
        """ Writes an image to provided file path. """

        if floc is None:
            self.Helpers.logger.info(
                "File location was not provided, file cannot be written.")
            return

        if data is None:
            self.Helpers.logger.info(
                "Image data was not provided, file cannot be written.")
            return

        try:
            cv2.imwrite(floc, data)
            print("File written! " + floc)
        except:
            print("File was not written! " + floc)
