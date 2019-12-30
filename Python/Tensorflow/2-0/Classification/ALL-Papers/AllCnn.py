############################################################################################
#
# Project:       Peter Moss Leukemia Research Foundation
# Repository:    ALL-IDB Classifiers
# Project:       Tensorflow 2.0 ALL Papers
#
# Author:        Adam Milton-Barker (adammiltonbarker@leukemiaresearchfoundation.ai)
# Contributors:
#
# Title:         AllCnn Wrapper Class
# Description:   Core AllCnn wrapper class for the Tensorflow 2.0 ALL Papers project.
# License:       MIT License
# Last Modified: 2019-12-30
#
############################################################################################

import sys

from Classes.Helpers import Helpers
from Classes.DataP1 import Data as DataP1
from Classes.ModelP1 import Model as ModelP1


class AllCnn():
    """ ALL Papers AllCnn Wrapper Class

    Core AllCnn wrapper class for the Tensorflow 2.0 ALL Papers project.
    """

    def __init__(self):

        self.Helpers = Helpers("Core")
        self.do_augmentation = False
        self.optimizer = ""

    def paper_1(self):
        """ Replicates the model proposed in Paper 1. 
        
        Replicates the networked and data splits outlined in the  Acute Leukemia Classification 
        Using Convolution Neural Network In Clinical Decision Support System paper
        using Tensorflow 2.0.

        https://airccj.org/CSCP/vol7/csit77505.pdf
        """
    
        self.DataP1 = DataP1(self.model_type, self.optimizer, self.do_augmentation)
        self.DataP1.data_and_labels_sort()
        
        if self.do_augmentation == False:
            self.DataP1.data_and_labels_prepare()
        else:
            self.DataP1.data_and_labels_augmentation_prepare()
        
        self.DataP1.shuffle()
        self.DataP1.get_split()

        self.ModelP1 = ModelP1(self.model_type, self.DataP1.X_train, self.DataP1.X_test, 
                               self.DataP1.y_train, self.DataP1.y_test, self.optimizer, self.do_augmentation)
        
        self.ModelP1.build_network()
        self.ModelP1.compile_and_train()
        
        self.ModelP1.save_model_as_json()
        self.ModelP1.save_weights()

        self.ModelP1.predictions()
        self.ModelP1.evaluate_model()
        self.ModelP1.plot_metrics()
        
        self.ModelP1.confusion_matrix()
        self.ModelP1.figures_of_merit()


AllCnn = AllCnn()


def main():
    
    if sys.argv[1] == "Adam":
        AllCnn.optimizer = "adam"
    elif sys.argv[1] == "RMSprop":
        AllCnn.optimizer = "rmsprop"
        
    if sys.argv[3] == 'True':
        AllCnn.do_augmentation = True
    else:
        AllCnn.do_augmentation = False
        
    if sys.argv[2] == '1':
        AllCnn.model_type = "model_1"
        AllCnn.paper_1()
    elif sys.argv[2] == '2':
        AllCnn.model_type = "model_2"
        print("Model 2 is currently not available yet")


if __name__ == "__main__":
    main()
