# Pill Detection Training
Files to help with pill detection. For more detailed instructions of training go [here](https://gist.github.com/douglasrizzo/c70e186678f126f1b9005ca83d8bd2ce) which is a summary of [this tutorial](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9).

1. [Install TensorFlow](https://www.tensorflow.org/install/)

2. Download / ```git clone``` the TensorFlow [models repository](https://github.com/tensorflow/models).

3. Download the current trained pill detection tensorflow graph .pb file [here](https://drive.google.com/file/d/1oyGktaQAoORLmCiX712Uy3RukjTI8_ws/view?usp=sharing)

## Create the dataset

1. Hand label pictures with [LabelImg](https://github.com/tzutalin/labelImg) and export images in the PASCAL VOC format which gives an XML file. 

2. Using the terminal, ```cd``` into the place where the XML is and run the ```xml_to_csv.py``` to convert the XML files from LabelImg into a single CSV file.

3. Separate the CSV file into two using the ```split labels.ipynb``` file. 

4. Convert the train and test CSVs to TFRecord files using the ```generate_tfrecord.py``` file.

## Training

1. Use ```image_processing/tf_model/mscoco_label_map.pbtxt``` as the label map.

2. Select a model to train [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
      
3. Provide a ```config``` file for the training pipeline. The one used is here ```image_processing/final_model/pipeline.config```

4. Train the model [locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md) or using [the cloud](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md).

5. Use [this to export]() the model and the resulting ```.pb``` file for the object detector.

## Testing

1. Use the ```image_processing/tf_model/detect_and_crop.py``` file to crop images based on detection boxes.

2. Use the cropped images with the [Siamese Network](https://github.com/mepotts/Pill-Siamese-Network/blob/master/siamese-network-pill-detect.ipynb).
