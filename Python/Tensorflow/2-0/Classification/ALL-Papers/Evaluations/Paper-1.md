# Peter Moss Leukemia Research Foundation

## ALL-IDB Classifiers

### ALL Papers Evaluation

&nbsp;

# Paper 1 Evaluation

Here we will replicate the network architecture and data split proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper and compare our results. In the paper, the authors do not go into the evaluation of the model, however, in this project we will go deeper into how well the model actually does. A useful tutorial while creating this network was the [Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) tutorial on Tensorflow's website.

&nbsp;

## ALL-IDB

You you need to be granted access to use the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. You can find the application form and information about getting access to the dataset on [this page](https://homes.di.unimi.it/scotti/all/#download) as well as information on how to contribute back to the project [here](https://homes.di.unimi.it/scotti/all/results.php). If you are not able to obtain a copy of the dataset please feel free to try this tutorial on your own dataset, we would be very happy to find additional AML & ALL datasets.

### ALL_IDB1 

In this paper, [this page](https://homes.di.unimi.it/scotti/all/#datasets) is used, one of the datsets from the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset.

"The ALL_IDB1 version 1.0 can be used both for testing segmentation capability of algorithms, as well as the classification systems and image preprocessing methods. This dataset is composed of 108 images collected during September, 2005. It contains about 39000 blood elements, where the lymphocytes has been labeled by expert oncologists. The images are taken with different magnifications of the microscope ranging from 300 to 500."  

In this project we will also use ALL_IDB1. The dataset is very small, with 108 examples.

&nbsp;

## Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System

<img src="https://www.leukemiaresearchfoundation.ai/github/media/images/paper_1_architecture.png" alt="Proposed Architecture" />

_Fig 1. Proposed architecture ([Source](https://airccj.org/CSCP/vol7/csit77505.pdf "Source"))_

In [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System"), the authors propose a simple 4 layer Convolutional Neural Network. 

> "In this work, we proposed a network contains 4 layers. The first 3 layers for detecting features
> and the other two layers (Fully connected and Softmax) are for classifying the features. The input
> image has the size [50x50x3]. The receptive field (or the filter size) is 5x5. The stride is 1 then we move the filters one pixel at a time. The zero-padding is 2. It will allow us to control the spatial
> size of the output image (we will use it to exactly preserve the spatial size of the input volume so
> the input and output width and height are the same). During the experiment, we found that in our
> case, altering the size of original image during the convolution lead to decrease the accuracy
> about 40%. Thus the output image after convolution layer 1 has the same size with the input
> image."

> "The convolution layer 2 has the same structure with the convolution layer 1. The filter size is 5x5,
> the stride is 1 and the zero-padding is 2. The number of feature maps (the channel or the depth) in
> our case is 30. If the number of feature maps is lower or higher than 30, the accuracy will
> decrease 50%. By experiment, we found the accuracy also decrease 50% if we remove
> Convolution layer 2.""

> "The Max-Pooling layer 25x25 has Filter size is 2 and stride is 2. The fully connected layer has 2
> neural. Finally, we use the Softmax layer for the classification. "

In this paper the authors used the ALL-IDB1 dataset, and did not use data augmentation to increase the training and testing data. However, in paper 2 the authors state that they had poor results using the model from paper 1 with augmented data. In my evaluation I use the dataset split proposed in paper 1, and the augmented dataset from paper 2, along with a custom network.

### Proposed Training / Validation Sets

In the paper the authors use the **ALL_IDB1** dataset. The paper proposes the following training and validation sets proposed in the paper, where **Normal cell** refers to ALL negative examples and **Abnormal cell** refers to ALL positive examples.

```
|               | Training Set | Test Set |
| ------------- | ------------ | -------- |
| Normal cell   | 40           | 19       |
| Abnormal cell | 40           | 9        |
| **Total**     | **80**       | **28**   |
```

### Architecture

We will build a Convolutional Neural Network, as shown in Fig 1, with an architecture consisting of the following 5 layers:

- Conv layer (50x50x30)
- Conv layer (50x50x30)
- Max-Pooling layer (25x25x30)
- Fully Connected layer (2 neurons)
- Softmax layer (Output 2)

&nbsp;

## Clone the repository

First of all you should clone the [ALL-IDB Classifiers](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers "ALL-IDB Classifiers") repository from the [Peter Moss Leukemia Research Foundation](https://github.com/LeukemiaResearchFoundation "Peter Moss Leukemia Research Foundation") Github Organization. 

To do this, make sure you have Git installed, navigate to the location you want to clone the repository to on your device using terminal/commandline, and then use the following command:

```
  $ git clone https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers.git
```

Once you have used the command above you will see a directory called **ALL-IDB-Classifiers** in the location you chose to clone to. In terminal, navigate to the **ALL-IDB-Classifiers/Python/Tensorflow/2-0/Classification/ALL-Papers/** directory, this is your project root directory.

&nbsp;

## Move the datasets

Now you need to move your ALL-IDB1 and ALL-IDB2 datasets to the **Model/Data** directory.

&nbsp;

## Configuration

[config.json](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Model/config.json "config.json")  holds the configuration for all of the networks that make up  the ALL Papers project. 

```
{
    "model_1": {
        "data": {
            "dim": 50,
            "file_type": ".jpg",
            "seed": 32,
            "train_dir": "Model/Data/ALL-IDB-1"
        },
        "train": {
            "batch": 80,
            "epochs": 75,
            "val_steps": 3
        }
    }
}
```

The configuration should be fairly self explanatory. We have the model_1 object containing two objects, data and train. In data we have the configuration related to preparing the training and validation data. We use a seed to make sure our results are reproducable. In train we have the configuration related to training the model.

Notice that the batch rate is 80, this is equal to the amount of data in the training data meaning that the network will see all samples in the dataset before updating the parmeters. This is done to try and reduce the spiking effect in our model's metrics. Other things that can help are batch normalization, more data and dropout, but in this project we are going to replicate the work proposed in the paper as close as possible. The seed is used to to ensure that the results should be easily replicated. 

In my case, the configuration above was the best out of my testing, but you may find different configurations work better. 
Feel free to update these settings to your liking, and please let us know of your experiences.

&nbsp;

## Code structure

The code for this project consists of 4 main Python files and a configuration file:

- [config.json](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Model/config.json "config.json"): The configuration file.
- [AllCnn.py](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/AllCnn.py "AllCnn.py"): A wrapper class.
- [Helpers.py](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Classes/Helpers.py "Helpers.py"): A helper class.
- [DataP1.py](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Classes/DataP1.py "DataP1.py"): A data helpers class.
- [ModelP1.py](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Classes/ModelP1.py "ModelP1.py"): A model helpers class.

&nbsp;

### Classes 

Our functionality for this network can be found mainly in the **Classes** directory. 

#### Helpers.py

[Helpers.py](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Classes/Helpers.py "Helpers.py") is a helper class for the Tensorflow 2.0 ALL Papers project. The class loads the configuration and logging that the project uses.

#### DataP1.py

[DataP1.py](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Classes/DataP1.py "DataP1.py") is a data helper class for the Paper 1 Evaluation. The class provides the functionality for sorting and preparing your training and validation data. the functions in this class reproduce the training and validation data split proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper.The main functions are briefly explained below:

##### data_and_labels_sort()

The data_and_labels_sort() function sorts the data into two Python lists, data[] and labels[].

##### data_and_labels_prepare()

The data_and_labels_sort() function prepares the data and labels for training, converting to Numpy arrays (np.array()).

##### convert_data()

The convert_data() function converts the training data to a numpy array.

##### encode_labels()

The encode_labels() function One Hot Encodes the labels.

##### shuffle()

The shuffle() function shuffles the data helping to eliminate bias.

##### get_split()

The get_split() function splits the prepared data and labels into traiing and validation data.

#### ModelP1.py

[ModelP1.py](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Classes/ModelP1.py "ModelP1.py") is a model helper class for the Paper 1 Evaluation. The class provides the functionality for creating our CNN. The main functions are briefly explained below:

##### build_network()

The build_network() function creates the network architecture proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper.

###### Metrics

We can use metrics to measure the effectiveness of our model. In this network we will use the following metrics:

```
tf.keras.metrics.BinaryAccuracy(name='accuracy'),
tf.keras.metrics.Precision(name='precision'),
tf.keras.metrics.Recall(name='recall'),
tf.keras.metrics.AUC(name='auc'),
tf.keras.metrics.TruePositives(name='tp'),
tf.keras.metrics.FalsePositives(name='fp'),
tf.keras.metrics.TrueNegatives(name='tn'),
tf.keras.metrics.FalseNegatives(name='fn') 
```

These metrics will be displayed and plotted once our model is trained.

##### compile_and_train()

The compile_and_train() function uses **model.compile()** and **model.fit()** to compile and train our model. For compiling we use the **tf.keras.optimizers.Adam()** optimizer with a learning rate of **1e-3**, and **binary crossentropy** as the loss function. 

model.fit() takes the training and test data, number of validation steps and number of epochs, and trains the model.

##### evaluate_model()

The evaluate_model() function evaluates the model, and displays the values for the metrics we specified.

&nbsp;


## Model Summary

Our network matches the architecture proposed in the paper exactly, with exception to the optimizer and loss function as this info was not provided in the paper.

```
Model: "AllCnn"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
zero_padding2d (ZeroPadding2 (None, 54, 54, 3)         0         
_________________________________________________________________
conv2d (Conv2D)              (None, 50, 50, 30)        2280      
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 54, 54, 30)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 50, 50, 30)        22530     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 25, 25, 30)        0         
_________________________________________________________________
flatten (Flatten)            (None, 18750)             0         
_________________________________________________________________
dense (Dense)                (None, 2)                 37502     
_________________________________________________________________
activation (Activation)      (None, 2)                 0         
=================================================================
Total params: 62,312
Trainable params: 62,312
Non-trainable params: 0
```

## Training the model

Now you are ready to train your model, make sure you are in the project root and execute the following command:

```
python3 AllCnn.py 1 0
```

This will start the classifier in Paper 1 basic mode, recreating the network and data split proposed in the paper. First the data functions are called, preparing our train and validation data. Next the model functions are called, creating the network, compiling and training the network, and finally evaluating the trained model.

```
Epoch 70/75
32/80 [===========>..................] - ETA: 0s - loss: 0.0752 - acc: 0.9688 - precision: 0.9688 - recall: 0.9688 - auc: 0.9971 - tp: 31.0000 - fp: 1.0000 - tn: 31.0000 - fn: 180/80 [==============================] - 0s 797us/sample - loss: 0.0445 - acc: 0.9875 - precision: 0.9875 - recall: 0.9875 - auc: 0.9992 - tp: 79.0000 - fp: 1.0000 - tn: 79.0000 - fn: 1.0000 - val_loss: 0.3601 - val_acc: 0.8571 - val_precision: 0.8571 - val_recall: 0.8571 - val_auc: 0.9554 - val_tp: 72.0000 - val_fp: 12.0000 - val_tn: 72.0000 - val_fn: 12.0000
Epoch 71/75
32/80 [===========>..................] - ETA: 0s - loss: 0.0621 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - tp: 32.0000 - fp: 0.0000e+00 - tn: 32.0000 - f80/80 [==============================] - 0s 796us/sample - loss: 0.0312 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - tp: 80.0000 - fp: 0.0000e+00 - tn: 80.0000 - fn: 0.0000e+00 - val_loss: 0.1556 - val_acc: 0.9643 - val_precision: 0.9643 - val_recall: 0.9643 - val_auc: 0.9885 - val_tp: 81.0000 - val_fp: 3.0000 - val_tn: 81.0000 - val_fn: 3.0000
Epoch 72/75
32/80 [===========>..................] - ETA: 0s - loss: 0.0201 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - tp: 32.0000 - fp: 0.0000e+00 - tn: 32.0000 - f80/80 [==============================] - 0s 793us/sample - loss: 0.0159 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - tp: 80.0000 - fp: 0.0000e+00 - tn: 80.0000 - fn: 0.0000e+00 - val_loss: 0.1405 - val_acc: 0.9643 - val_precision: 0.9643 - val_recall: 0.9643 - val_auc: 0.9911 - val_tp: 81.0000 - val_fp: 3.0000 - val_tn: 81.0000 - val_fn: 3.0000
Epoch 73/75
32/80 [===========>..................] - ETA: 0s - loss: 0.0190 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - tp: 32.0000 - fp: 0.0000e+00 - tn: 32.0000 - f80/80 [==============================] - 0s 791us/sample - loss: 0.0168 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - tp: 80.0000 - fp: 0.0000e+00 - tn: 80.0000 - fn: 0.0000e+00 - val_loss: 0.2682 - val_acc: 0.8929 - val_precision: 0.8929 - val_recall: 0.8929 - val_auc: 0.9719 - val_tp: 75.0000 - val_fp: 9.0000 - val_tn: 75.0000 - val_fn: 9.0000
Epoch 74/75
32/80 [===========>..................] - ETA: 0s - loss: 0.0100 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - tp: 32.0000 - fp: 0.0000e+00 - tn: 32.0000 - f80/80 [==============================] - 0s 786us/sample - loss: 0.0179 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - tp: 80.0000 - fp: 0.0000e+00 - tn: 80.0000 - fn: 0.0000e+00 - val_loss: 0.2188 - val_acc: 0.9286 - val_precision: 0.9286 - val_recall: 0.9286 - val_auc: 0.9821 - val_tp: 78.0000 - val_fp: 6.0000 - val_tn: 78.0000 - val_fn: 6.0000
Epoch 75/75
32/80 [===========>..................] - ETA: 0s - loss: 0.0063 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - tp: 32.0000 - fp: 0.0000e+00 - tn: 32.0000 - f80/80 [==============================] - 0s 796us/sample - loss: 0.0152 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - tp: 80.0000 - fp: 0.0000e+00 - tn: 80.0000 - fn: 0.0000e+00 - val_loss: 0.1846 - val_acc: 0.9286 - val_precision: 0.9286 - val_recall: 0.9286 - val_auc: 0.9872 - val_tp: 78.0000 - val_fp: 6.0000 - val_tn: 78.0000 - val_fn: 6.0000
<tensorflow.python.keras.callbacks.History object at 0x7f485c520c18>
```

### Model evaluation

<img src="../Model/metrics.png" alt="Model evaluation metrics" />

```
2019-12-28 06:33:24,799 - Model - INFO - Metrics: loss 0.1846487820148468
2019-12-28 06:33:24,799 - Model - INFO - Metrics: acc 0.9285714
2019-12-28 06:33:24,799 - Model - INFO - Metrics: precision 0.9285714
2019-12-28 06:33:24,799 - Model - INFO - Metrics: recall 0.9285714
2019-12-28 06:33:24,799 - Model - INFO - Metrics: auc 0.98724496
2019-12-28 06:33:24,799 - Model - INFO - Metrics: tp 26.0
2019-12-28 06:33:24,799 - Model - INFO - Metrics: fp 2.0
2019-12-28 06:33:24,799 - Model - INFO - Metrics: tn 26.0
2019-12-28 06:33:24,799 - Model - INFO - Metrics: fn 2.0
```

&nbsp;

## Obeservations

We can notice that metrics for Accuracy, Precision & Recall are the same, this could be due to the small size of the dataset. In the second part of this evaluation we will augment the data using the methods proposed in Paper 2, [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon, to see if the increase in data corrects this issue. We can also notice a spiking effect in the metric plots, this could also be related to the size of the dataset. In the next part of this evaluation we will also experiment with batch normalization and dropout to see how it helps our results. There is more tweaking that can be done on this network, but for now this is a good start.

&nbsp;

## Results (75 epochs)

| Loss          | Accuracy     | Precision     | Recall       | AUC          |
| ------------- | ------------ | ------------- | ------------ | ------------ |
| 0.131 (~0.13) | 0.929 (~93%) | 0.929 (~0.93%) | 0.929 (~93%) | 0.984 (~1.0) |

&nbsp;

## ALL-IDB figures of merit

On the report the [results page](https://homes.di.unimi.it/scotti/all/results.php) on the ALL-IDB website, you can find info about suggested reporting for projects using the ALL-IDB dataset. Once your model has finished training, these stats will be displayed. Please note you have to close the metrics image before the program can complete. 

```
2019-12-28 06:34:08,792 - Model - INFO - True Positives: 11(39.285714285714285%)
2019-12-28 06:34:08,792 - Model - INFO - False Positives: 0(0.0%)
2019-12-28 06:34:08,793 - Model - INFO - True Negatives: 15(53.57142857142857%)
2019-12-28 06:34:08,793 - Model - INFO - False Negatives: 2(7.142857142857143%)
2019-12-28 06:34:08,793 - Model - INFO - Specificity: 1.0
2019-12-28 06:34:08,793 - Model - INFO - Misclassification: 2(7.142857142857143%)
```

| Figures of merit     | Value | Percentage |
| -------------------- | ----- | ---------- |
| True Positives       | 11    | 39.30%     |
| False Positives      | 0     | 0.00%      |
| True Negatives       | 15    | 0.00%      |
| False Negatives      | 2     | 7.14%      |
| Misclassification    | 2     | 7.14%      |
| Sensitivity / Recall | 0.929 | 93%        |
| Specificity          | 1     | 100%       |

&nbsp;

# Contributing

The Peter Moss Leukemia Research Foundation & Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourage and welcome code contributions, bug fixes and enhancements from the Github community.

Please read the [CONTRIBUTING](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

&nbsp;

# Bugs/Issues

We use the [repo issues](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/issues "repo issues") to track bugs and general requests related to using this project.

&nbsp;

# Versioning

We use SemVer for versioning. For the versions available, see [Releases](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/releases "Releases").

&nbsp;

# Authors

- [Adam Milton-Barker](https://www.leukemiaresearchfoundation.ai/team/adam-milton-barker/profile "Adam Milton-Barker") - Peter Moss Leukemia Research Foundation Founder & Intel Software Innovator, Barcelona, Spain

See full list of [contributors](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/contributors "contributors") that were involved in this project.

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/LeukemiaResearchFoundation/ALL-IDB-Classifiers/blob/master/LICENSE.md "LICENSE") file for details.