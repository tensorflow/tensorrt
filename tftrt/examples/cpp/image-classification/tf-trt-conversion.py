#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# Copyright 2019 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Notebook for converting native Tensorflow frozen graph to TF-TRT model

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


config = tf.ConfigProto()
config.gpu_options.allow_growth=True


# %%


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def load_graph_def(model_file):  
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  
  return graph_def


# %%


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  result = sess.run(normalized)

  return result


# %%


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


# %%


file_name = "./data/grace_hopper.jpg"
model_file = "./data/inception_v3_2016_08_28_frozen.pb"
label_file = "./data/imagenet_slim_labels.txt"
input_height = 299
input_width = 299
input_mean = 0
input_std = 255
input_layer = "input"
output_layer = "InceptionV3/Predictions/Reshape_1"


# %%


graph = load_graph(model_file)
    
t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)


# %%


input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

with tf.compat.v1.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
results = np.squeeze(results)

top_k = results.argsort()[-5:][::-1]
labels = load_labels(label_file)
for i in top_k:
    print(labels[i], results[i])


# %%


# Benchmark native TensorFlow model

N_warmup_run = 50
N_run = 1000
elapsed_time = []
BATCH_SIZE = 1 

print("Benchmark native TensorFlow model...")
with tf.compat.v1.Session(graph=graph) as sess:
    for i in range(N_warmup_run):
        results = sess.run(output_operation.outputs[0], {
                           input_operation.outputs[0]: t
                           })

    for i in range(N_run):
      start_time = time.time()
      results = sess.run(output_operation.outputs[0], {
                           input_operation.outputs[0]: t
                           })
      end_time = time.time()
      elapsed_time = np.append(elapsed_time, end_time - start_time)
      if i % 50 == 0:
        print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_run * BATCH_SIZE / elapsed_time.sum()))


# %%


BATCH_SIZE = 1

graph_def = load_graph_def(model_file)
    
trt_fp32_graph = trt.create_inference_graph(
    input_graph_def=graph_def,
    outputs=['InceptionV3/Predictions/Reshape_1'],
    max_batch_size=BATCH_SIZE,
    precision_mode="FP32")


# %%


with tf.gfile.GFile('./data/inception_v3_2016_08_28_frozen_tftrt_fp32.pb', 'wb') as f:
    f.write(trt_fp32_graph.SerializeToString())
    
print("Successfully export TF-TRT model to ./data/inception_v3_2016_08_28_frozen_tftrt_fp32.pb")


# %%


# Testing TF-TRT Model
graph = load_graph('./data/inception_v3_2016_08_28_frozen_tftrt_fp32.pb')

input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

with tf.compat.v1.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
results = np.squeeze(results)

top_k = results.argsort()[-5:][::-1]
labels = load_labels(label_file)
for i in top_k:
    print(labels[i], results[i])


# %%


# Benchmark TF-TRT model

N_warmup_run = 50
N_run = 1000
elapsed_time = []
batch_size = 1 

print("Benchmark TF-TRT model...")
with tf.compat.v1.Session(graph=graph) as sess:
    for i in range(N_warmup_run):
        results = sess.run(output_operation.outputs[0], {
                           input_operation.outputs[0]: t
                           })

    for i in range(N_run):
      start_time = time.time()
      results = sess.run(output_operation.outputs[0], {
                           input_operation.outputs[0]: t
                           })
      end_time = time.time()
      elapsed_time = np.append(elapsed_time, end_time - start_time)
      if i % 50 == 0:
        print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_run * BATCH_SIZE / elapsed_time.sum()))


# %%




