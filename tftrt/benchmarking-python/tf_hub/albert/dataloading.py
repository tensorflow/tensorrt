import tensorflow as tf
import csv
import tensorflow_datasets as tfds
import tensorflow_hub as tf_hub
import tensorflow_text as text
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))


def read_tsv(input_file):
    """Reads a tab separated value file."""
    with tf.io.gfile.GFile(input_file, 'r') as f:
      reader = csv.reader(f, delimiter="\t", quotechar='|')
      lines = []
      for line in reader:
        lines.append(line)
      return lines


def get_cola_labels():
    return ["0","1"]

def get_dataset_cola(sequence_length, batch_size,
        vocab_size, full_path_to_file='test_cola.tsv'):
     """Loads the data from the specified tsv file."""
     import tensorflow_hub as hub

     lines = read_tsv(full_path_to_file)
     preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_preprocess/3")
     ds_mask = []
     ds_typeids = []
     ds_wordids = []
     for (i, line) in enumerate(lines):
        encoder_ips =  preprocessor(line)
        # dict_keys(['input_word_ids', 'input_type_ids', 'input_mask'])
        ds_mask.append(encoder_ips["input_mask"])
        ds_typeids.append(encoder_ips["input_type_ids"])
        ds_wordids.append(encoder_ips["input_word_ids"])

     ds_mask = tf.data.Dataset.from_tensor_slices(input_mask)
     ds_typeids = tf.data.Dataset.from_tensor_slices(guid_arr)
     ds_wordids = tf.data.Dataset.from_tensor_slices(text_arr)
     dataset = tf.data.Dataset.zip((ds_mask, ds_typeids, ds_wordids))

     # dataset = dataset.repeat()
     dataset = dataset.batch(batch_size, drop_remainder=False)
     #dataset = dataset.take(count=1)  # loop over 1 batch
     #dataset = dataset.cache()
     #dataset = dataset.repeat()
     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

     return dataset
