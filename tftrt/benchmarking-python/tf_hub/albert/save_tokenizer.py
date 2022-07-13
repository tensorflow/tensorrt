import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Registers the ops.


def save_module(url, save_path):
  module = hub.KerasLayer(url)
  model = tf.keras.Sequential(module)
  tf.saved_model.save(model, save_path)


save_module(
    url="https://tfhub.dev/tensorflow/albert_en_preprocess/3",
    save_path="/models/tf_hub/albert/tokenizer"
)
