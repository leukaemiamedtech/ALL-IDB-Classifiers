############################################################################################
#
# Project:       Peter Moss Leukemia Research Foundation
# Repository:    ALL-IDB Classifiers
# Project:       Tensorflow 2.0 ALL Papers
#
# Author:        Adam Milton-Barker (adammiltonbarker@leukemiaresearchfoundation.ai)
# Contributors:
#
# Title:         Paper 1 Model Helper Class
# Description:   Model helper class for the Paper 1 Evaluation.
# License:       MIT License
# Last Modified: 2019-12-28
#
############################################################################################

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

from Classes.Helpers import Helpers


class Model():
    """ Model Class
    
    Model helper class for the Paper 1 Evaluation.
    """

    def __init__(self, model):
        """ Initializes the Model class. """

        self.Helpers = Helpers("Model", False)
        self.model_type = model
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.Helpers.logger.info("Model class initialization complete.")

    def build_network(self, isize):
        """ Creates the Paper 1 Evaluation network. 
        
        Replicates the networked outlined in the  Acute Leukemia Classification 
        Using Convolution Neural Network In Clinical Decision Support System paper
        using Tensorflow 2.0.

        https://airccj.org/CSCP/vol7/csit77505.pdf
        """

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.ZeroPadding2D(
                padding=(2, 2), input_shape=isize),
            tf.keras.layers.Conv2D(30, (5, 5), strides=1,
                                   padding="valid", activation='relu'),
            tf.keras.layers.ZeroPadding2D(padding=(2, 2)),
            tf.keras.layers.Conv2D(30, (5, 5), strides=1,
                                   padding="valid", activation='relu'),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
            tf.keras.layers.Activation('softmax')
        ], 
        "AllCnn")
        self.model.summary()
        self.Helpers.logger.info("Network built")

    def compile_and_train(self, X_train, X_test, y_train, y_test):
        """ Compiles the Paper 1 Evaluation model. """

        optimizer =  tf.keras.optimizers.Adam(lr=1e-3)
        #optimizer =  tf.keras.optimizers.Adam()
        
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=[tf.keras.metrics.BinaryAccuracy(name='acc'),
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),
                                    tf.keras.metrics.AUC(name='auc'),
                                    tf.keras.metrics.TruePositives(name='tp'),
                                    tf.keras.metrics.FalsePositives(name='fp'),
                                    tf.keras.metrics.TrueNegatives(name='tn'),
                                    tf.keras.metrics.FalseNegatives(name='fn') ])

        self.history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), validation_steps=self.Helpers.confs[
                                          self.model_type]["train"]["val_steps"], epochs=self.Helpers.confs[self.model_type]["train"]["epochs"])

        print(self.history)
        print("") 
    
    def predictions(self, X_train, X_test):
        """ Makes predictions on the test set. """
        
        self.train_preds = self.model.predict(X_train)
        self.test_preds = self.model.predict(X_test)
        self.Helpers.logger.info("Training predictions: " + str(self.train_preds))
        self.Helpers.logger.info("Testing predictions: " + str(self.test_preds))
        print("")

    def evaluate_model(self, X_test, y_test, y_train ):
        """ Evaluates the Paper 1 Evaluation model. """
        
        metrics = self.model.evaluate(X_test, y_test, verbose=0)        
        for name, value in zip(self.model.metrics_names, metrics):
            self.Helpers.logger.info("Metrics: " + name + " " + str(value))
        print()
        
        plt.legend(loc='lower right')
        
    def plot_metrics(self):
        """ Plots our metrics. 
        
        https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        """
                
        metrics =  ['acc', 'loss', 'auc', 'precision', 'recall', 'tn']
        for n, metric in enumerate(metrics):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(2,3,n+1)
            plt.plot(self.history.epoch,  self.history.history[metric], color=self.colors[0], label='Train')
            plt.plot(self.history.epoch, self.history.history['val_'+metric],
                    color=self.colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8,1])
            else:
                plt.ylim([0,1])

            plt.legend()
        
    def confusion_matrix(self, y_test):
        
        self.matrix = confusion_matrix(y_test.argmax(axis=1), 
                                       self.test_preds.argmax(axis=1))
        
        self.Helpers.logger.info("Confusion Matrix: " + str(self.matrix))
        print("")
        
        plt.imshow(self.matrix, cmap=plt.cm.Blues)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Confusion matrix ')
        plt.colorbar()
        plt.show()
            
    def figures_of_merit(self, X_test):
        
        test_len = len(X_test)
        
        TP = self.matrix[1][1]
        TN = self.matrix[0][0]
        FP = self.matrix[0][1]
        FN = self.matrix[1][0]
        
        TPP = (TP * 100)/test_len
        FPP = (FP * 100)/test_len
        FNP = (FN * 100)/test_len
        TNP = (TN * 100)/test_len
        
        self.Helpers.logger.info("True Positives: " + str(TP) + "(" + str(TPP) + "%)")
        self.Helpers.logger.info("False Positives: " + str(FP) + "(" + str(FPP) + "%)")
        self.Helpers.logger.info("True Negatives: " + str(TN) + "(" + str(TNP) + "%)")
        self.Helpers.logger.info("False Negatives: " + str(FN) + "(" + str(FNP) + "%)")
        
        specificity = TN/(TN+FP) 
        self.Helpers.logger.info("Specificity: " + str(specificity))
        
        misc = FP + FN        
        miscp = (misc * 100)/test_len 
        self.Helpers.logger.info("Misclassification: " + str(misc) + "(" + str(miscp) + "%)")        
        
    def save_weights(self):
        """ Saves the model weights. """
            
        self.model.save_weights("Model/weights.h5")  
        
    def save_model_as_json(self):
        """ Saves the model to JSON. """
        
        with open("Model/model.json", "w") as file:
            file.write(self.model.to_json())
        
    def load_model_from_json(self):
        """ Loads the model from JSON. """
        
        with open("Model/model.json", "w") as file:
            jmodel = file.read()
        
        tf.Keras.set_learning_phase(0)
        model = tf.keras.models.model_from_json(jmodel) 
        model.load_weights_model("Model/weights.h5")
        
        model.summary() 