
# Prerequisite: Install the python module below before running this example.
# pip install -q tf-models-official

import tensorflow as tf
import tensorflow_hub as hub

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
bert_saved_model_path = './models/bert_base'

bert_model = hub.load(tfhub_handle_encoder)
tf.saved_model.save(bert_model, bert_saved_model_path)

import numpy as np
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Instantiate the TF-TRT converter
PROFILE_STRATEGY="Optimal"
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=bert_saved_model_path,
    precision_mode=trt.TrtPrecisionMode.FP32,
    use_dynamic_shape=True,
    dynamic_shape_profile_strategy=PROFILE_STRATEGY)

# Convert the model to TF-TRT
converter.convert()

VOCAB_SIZE = 30522  # Model specific, look in the model README.
# Build engines for input sequence lengths of 128, and 384.
input_shapes = [[(1, 128), (1, 128), (1, 128)],
                [(1, 384), (1, 384), (1, 384)]]
def input_fn():
    for shapes in input_shapes:
        # return a list of input tensors
        yield [tf.convert_to_tensor(
          np.random.randint(low=0, high=VOCAB_SIZE, size=x,dtype=np.int32))
          for x in shapes]

converter.build(input_fn)

# Save the converted model
bert_trt_path = "./models/tftrt_bert_base"
converter.save(bert_trt_path)
converter.summary()

# Some helper functions
def get_func_from_saved_model(saved_model_dir):
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return graph_func, saved_model_loaded


def get_random_input(batch_size, seq_length):
    # Generate random input data
    mask = tf.convert_to_tensor(np.ones((batch_size, seq_length), dtype=np.int32))
    type_id = tf.convert_to_tensor(np.zeros((batch_size, seq_length), dtype=np.int32))
    word_id = tf.convert_to_tensor(
      np.random.randint(0, VOCAB_SIZE, size=[batch_size, seq_length], dtype=np.int32))
    return {'input_mask':mask, 'input_type_ids': type_id, 'input_word_ids':word_id}

# Get a random input tensor
input_tensor = get_random_input(1, 128)

# Specify the output tensor interested in. This output is the 'classifier'
result_key = 'bert_encoder_1'
trt_func, _ = get_func_from_saved_model(bert_trt_path)

## Let's run some inferences!
for i in range(0, 10):
    print(f"Step: {i}")
    preds = trt_func(**input_tensor)
    result = preds[result_key]
