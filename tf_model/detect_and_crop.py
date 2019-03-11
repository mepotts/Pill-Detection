#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:37:08 2019

@author: katepotts
"""

# from distutils.version import StrictVersion
import os, sys,tarfile, zipfile
import numpy as np
import tensorflow as tf
import six.moves.urllib as urllib
from PIL import Image
from io import StringIO
from matplotlib import pyplot as plt
from collections import defaultdict
from object_detection.utils import ops as utils_ops
from protos import string_int_label_map_pb2
# Paths settings
THE_PATH = "/Users/katepotts/anaconda3/lib/python3.7/site-packages/tensorflow/models/research"
sys.path.append(THE_PATH)
sys.path.append(THE_PATH+"/object_detection")
PATH_TO_FROZEN_GRAPH = "/Users/katepotts/dataset/final_model/frozen_inference_graph.pb"
PATH_TO_LABELS = "/Users/katepotts/dataset/data/mscoco_label_map.pbtxt"
PATH_TO_TEST_IMAGES_DIR = '/Users/katepotts/Downloads/dc'
PATH_TO_CROP_IMAGES_DIR = '/Users/katepotts/dc_crop'

l=os.listdir(PATH_TO_TEST_IMAGES_DIR)
li=[x.split('.')[0] for x in l]
str_list = list(filter(None, li))

TEST_IMAGE_NAMES = str_list
filetype = '.jpg'
TEST_IMAGE_PATHS = [''.join((PATH_TO_TEST_IMAGES_DIR, '/', x, filetype)) for x in TEST_IMAGE_NAMES]
# print("test image path = ", TEST_IMAGE_PATHS)
IMAGE_SIZE = (12, 8) # Size, in inches, of the output images.
NUM_CLASSES = 1

from utils import label_map_util
from utils import visualization_utils as vis_util
sys.path.append("..")
# MODEL_NAME = 'faster_rcnn_resnet101_pets'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map  = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#print(category_index)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

for image_path, image_name in zip(TEST_IMAGE_PATHS, TEST_IMAGE_NAMES):
  print(image_path)
    
  image = Image.open(image_path)
  
  im_width, im_height = image.size
  
  #if im_width < im_height:
  #    image = image.rotate(-90)
  #else:
  #    image
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      # ercx!
      # each element in DETECTION BOX is [ymin, xmin, ymax, xmax]
      # do consider the following
      #           im_width, im_height = image.size
      #       if use_normalized_coordinates:
      #         (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
      #                                       ymin * im_height, ymax * im_height)
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=2,
      min_score_thresh = 0.01)
  # print("detection_boxes:")
  # print(output_dict['detection_boxes'])
  # print(type(output_dict['detection_boxes']),len(output_dict['detection_boxes']))
  # print('detection_classes')
  # print(output_dict['detection_classes'])
  # print(type(output_dict['detection_classes']),len(output_dict['detection_classes']))
  # print('detection_scores')
  # print(output_dict['detection_scores'], len(output_dict['detection_scores']))
  plt.imsave(''.join((PATH_TO_CROP_IMAGES_DIR, '//',image_name,"_MARKED", filetype)), image_np)
  
  try:
      ymin = output_dict['detection_boxes'][0][0]
      xmin = output_dict['detection_boxes'][0][1]
      ymax = output_dict['detection_boxes'][0][2]
      xmax = output_dict['detection_boxes'][0][3]
  
      (xminn, xmaxx, yminn, ymaxx) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
      area = (xminn, yminn, xmaxx, ymaxx)
      cropped_image = image.crop(area)
      cropped_image.save(''.join((PATH_TO_CROP_IMAGES_DIR, '/',image_name,"_CROPPED", filetype)))
  except Exception:
      pass
  print('\n**************** detection_scores\n')
  print(output_dict['detection_scores'][1:10])
  #plt.figure(figsize=IMAGE_SIZE)
  # plt.imshow(image_np)
  
  
