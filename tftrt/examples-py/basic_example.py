import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.python.framework import ops

tf.debugging.set_log_device_placement(True) 
tf.get_logger().setLevel('INFO')

INPUT_SIZE = (512, 5000)
SAVED_MODEL_DIR="/tmp/1234"


def extract_devices_from_graphdef(graphdef):
    all_nodes = [n for n in graphdef.node]
    all_devices = list(set([n.device for n in all_nodes]))
    return all_devices


def create_model():
  """Define a simple sequential model"""
  model = tf.keras.models.Sequential([
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2048, activation='sigmoid'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1024, activation='softsign'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation='softmax'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation='softplus'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation=tf.nn.log_softmax),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='selu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='elu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])
  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])
  return model


if __name__ == "__main__":

    # Create a basic model instance
    model = create_model()
    
    # Build model
    _ = model(tf.random.uniform(INPUT_SIZE))

    # Save Model
    model.save(SAVED_MODEL_DIR)

    if False:
        from tensorflow.python.saved_model import signature_constants
        from tensorflow.python.saved_model import tag_constants

        tags = [tag_constants.SERVING]
        signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        saved_model_loaded = tf.saved_model.load(export_dir=path, tags=tags)
        func = saved_model_loaded.signatures[signature_key]

    else:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt

        # Conversion Parameters 
        conversion_params = trt.TrtConversionParams(
            precision_mode=trt.TrtPrecisionMode.FP32
        )

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=SAVED_MODEL_DIR,
            conversion_params=conversion_params
        )

        # Converter method used to partition and optimize TensorRT compatible segments
        func = converter.convert()
    
    converter.summary()

    data = tf.random.uniform(INPUT_SIZE)
    for step in range(1, 10):
      if step % 50 == 0:
        print(f"Step: {step}")
      func(data)

    for node in func.graph.as_graph_def().node:
      print(f"- OP: {node.op} - Name: {node.name} - Device: {node.device}")

