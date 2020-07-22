############################################################################################
#
# Project:       Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss
# Repository:    ALL-IDB Classifiers
# Project:       Paper 1
#
# Author:        Adam Milton-Barker
# Contributors:
#
# Title:         Helpers Class
# Description:   Helpers class for the Paper 1 Evaluation.
# License:       MIT License
# Last Modified: 2019-07-23
#
############################################################################################

import json, logging, sys, time
import logging.handlers as handlers

from datetime import datetime


class Helpers():
    """ Paper 1 Helper Class

    Helper class for the Tensorflow 2.0 ALL Papers project.
    """

    def __init__(self, ltype, log=True):
        """ Initializes the Helpers Class. """

        self.confs = {}
        self.loadConfs()

        self.logger = logging.getLogger(ltype)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        allLogHandler = handlers.TimedRotatingFileHandler(
            'Logs/all.log', when='H', interval=1, backupCount=0)
        allLogHandler.setLevel(logging.INFO)
        allLogHandler.setFormatter(formatter)

        errorLogHandler = handlers.TimedRotatingFileHandler(
            'Logs/error.log', when='H', interval=1, backupCount=0)
        errorLogHandler.setLevel(logging.ERROR)
        errorLogHandler.setFormatter(formatter)

        warningLogHandler = handlers.TimedRotatingFileHandler(
            'Logs/warning.log', when='H', interval=1, backupCount=0)
        warningLogHandler.setLevel(logging.WARNING)
        warningLogHandler.setFormatter(formatter)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(formatter)

        self.logger.addHandler(allLogHandler)
        self.logger.addHandler(errorLogHandler)
        self.logger.addHandler(warningLogHandler)
        self.logger.addHandler(consoleHandler)

        if log is True:
            self.logger.info("Helpers class initialization complete.")

    def loadConfs(self):
        """ Load the program configuration. """

        with open('Model/config.json') as confs:
            self.confs = json.loads(confs.read())
