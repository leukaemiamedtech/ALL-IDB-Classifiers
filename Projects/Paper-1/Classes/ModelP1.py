############################################################################################
#
# Project:       Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss
# Repository:    ALL-IDB Classifiers
# Project:       Paper 1
#
# Author:        Adam Milton-Barker
# Contributors:
#
# Title:         Model Class
# Description:   Model helper class for the Paper 1 Evaluation.
# License:       MIT License
# Last Modified: 2019-07-23
#
############################################################################################

import random

import tensorflow as tf
import matplotlib.pyplot as plt

from numpy.random import seed
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

from Classes.Helpers import Helpers


class Model():
    """ Model Class
    
    Model helper class for the Paper 1 Evaluation.
    """

    def __init__(self, model, X_train,  X_test, y_train, 
                 y_test, optimizer, do_augmentation = False):
        """ Initializes the Model class. """

        self.Helpers = Helpers("Model", False)
        self.model_type = model
        self.optimizer = optimizer
        
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        if do_augmentation == False:
            self.seed = self.Helpers.confs[self.model_type]["data"]["seed_" + self.optimizer]
            self.val_steps = self.Helpers.confs[self.model_type]["train"]["val_steps"]
            self.batch_size = self.Helpers.confs[self.model_type]["train"]["batch"] 
            self.epochs = self.Helpers.confs[self.model_type]["train"]["epochs"]
            self.weights_file = "Model/weights.h5"
            self.model_json = "Model/model.json"
        else:
            self.seed = self.Helpers.confs[self.model_type]["data"]["seed_" + self.optimizer + "_augmentation"]
            self.val_steps = self.Helpers.confs[self.model_type]["train"]["val_steps_augmentation"]
            self.batch_size = self.Helpers.confs[self.model_type]["train"]["batch_augmentation"] 
            self.epochs = self.Helpers.confs[self.model_type]["train"]["epochs_augmentation"]
            self.weights_file = "Model/weights_augmentation.h5"
            self.model_json = "Model/model_augmentation.json"
            
        random.seed(self.seed)
        seed(self.seed)
        tf.random.set_seed(self.seed)
            
        self.Helpers.logger.info("Model class initialization complete.")

    def build_network(self):
        """ Creates the Paper 1 Evaluation network. 
        
        Replicates the networked outlined in the  Acute Leukemia Classification 
        Using Convolution Neural Network In Clinical Decision Support System paper
        using Tensorflow 2.0.

        https://airccj.org/CSCP/vol7/csit77505.pdf
        """

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.ZeroPadding2D(
                padding=(2, 2), input_shape=self.X_train.shape[1:]),
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

    def compile_and_train(self):
        """ Compiles the Paper 1 Evaluation model. """

        if self.optimizer == "adam":            
            self.Helpers.logger.info("Using Adam Optimizer.")
            optimizer =  tf.keras.optimizers.Adam(lr=self.Helpers.confs[self.model_type]["train"]["learning_rate_adam"], 
                                                  decay = self.Helpers.confs[self.model_type]["train"]["decay_adam"])
            #optimizer =  tf.keras.optimizers.Adam()
        else:
            self.Helpers.logger.info("Using RMSprop Optimizer.")
            optimizer = tf.keras.optimizers.RMSprop(lr = self.Helpers.confs[self.model_type]["train"]["learning_rate_rmsprop"], 
                                                    decay = self.Helpers.confs[self.model_type]["train"]["decay_rmsprop"])
        
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

        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), 
                                      validation_steps=self.val_steps, epochs=self.epochs)

        print(self.history)
        print("") 
    
    def predictions(self):
        """ Makes predictions on the test set. """
        
        self.train_preds = self.model.predict(self.X_train)
        self.test_preds = self.model.predict(self.X_test)
        
        self.Helpers.logger.info("Training predictions: " + str(self.train_preds))
        self.Helpers.logger.info("Testing predictions: " + str(self.test_preds))
        print("")

    def evaluate_model(self):
        """ Evaluates the Paper 1 Evaluation model. """
        
        metrics = self.model.evaluate(self.X_test, self.y_test, verbose=0)        
        for name, value in zip(self.model.metrics_names, metrics):
            self.Helpers.logger.info("Metrics: " + name + " " + str(value))
        print()
        
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
        
    def confusion_matrix(self):
        """ Prints/displays the confusion matrix. """
        
        self.matrix = confusion_matrix(self.y_test.argmax(axis=1), 
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
            
    def figures_of_merit(self):
        """ Calculates/prints the figures of merit. 
        
        https://homes.di.unimi.it/scotti/all/
        """
        
        test_len = len(self.X_test)
        
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
            
        self.model.save_weights(self.weights_file)  
        
    def save_model_as_json(self):
        """ Saves the model to JSON. """
        
        with open(self.model_json, "w") as file:
            file.write(self.model.to_json())
        
    def load_model_from_json(self):
        """ Loads the model from JSON. """
        
        with open(self.model_json, "w") as file:
            json_model = file.read()
        
        tf.keras.set_learning_phase(0)
        model = tf.keras.models.model_from_json(json_model) 
        model.load_weights_model(self.weights_file)
        
        model.summary() 