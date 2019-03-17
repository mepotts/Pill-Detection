# Pill Detection Training
files to help with pill detection

1. [Install TensorFlow](https://www.tensorflow.org/install/)

2. Download the TensorFlow `models repository <https://github.com/tensorflow/models>`__.

## Create the dataset

1. Hand label pictures with [LabelImg](https://github.com/tzutalin/labelImg) and export images in the PASCAL VOC format which gives an XML file. 

2. Using the terminal, ```cd``` into the place where the XML is and run the ```xml_to_csv.py```
