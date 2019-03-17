# Pill Detection Training
files to help with pill detection

1. [Install TensorFlow](https://www.tensorflow.org/install/)

2. Download the TensorFlow [models repository](https://github.com/tensorflow/models).

## Create the dataset

1. Hand label pictures with [LabelImg](https://github.com/tzutalin/labelImg) and export images in the PASCAL VOC format which gives an XML file. 

2. Using the terminal, ```cd``` into the place where the XML is and run the ```xml_to_csv.py``` to convert the XML files from LabelImg into a single CSV file.

3. Separate the CSV file into two using the ```split labels.ipynb``` file. 

4. Convert the train and test CSVs to TFRecord files using the ```generate_tfrecord.py``` file.

## Training

1. Use ```image_processing/tf_model/mscoco_label_map.pbtxt``` as the label map.

2. Select a model to train [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
      
