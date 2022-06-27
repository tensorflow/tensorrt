import tensorflow as tf
import csv
import tensorflow_datasets as tfds


def get_dataset(sequence_length=128, batch_size=32, vocab_size=32000):
    tf.random.set_seed(10)

    input_mask = tf.random.uniform(
        shape=(1, sequence_length),
        maxval=vocab_size,
        dtype=tf.int32
    )
    input_type_ids = tf.random.uniform(
        shape=(1, sequence_length),
        maxval=vocab_size,
        dtype=tf.int32
    )
    input_word_ids = tf.random.uniform(
        shape=(1, sequence_length),
        maxval=vocab_size,
        dtype=tf.int32
    )

    ds_mask = tf.data.Dataset.from_tensor_slices(input_mask)
    ds_typeids = tf.data.Dataset.from_tensor_slices(input_type_ids)
    ds_wordids = tf.data.Dataset.from_tensor_slices(input_word_ids)
    dataset = tf.data.Dataset.zip((ds_mask, ds_typeids, ds_wordids))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.take(count=1)  # loop over 1 batch
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

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

def get_dataset_cola(sequence_length=128, batch_size=32,
        vocab_size=32000, full_path_to_file='test_cola.tsv'):
     """Loads the data from the specified tsv file."""
     from transformers import BertTokenizer

     tz = BertTokenizer.from_pretrained("bert-base-cased")
     lines = read_tsv(full_path_to_file)
     guid_arr = []
     text_arr = []
     set_type = 'test'
     for (i,line) in enumerate(lines):
        # print("Working on line: {} with length {} \n".format(i,len(line)))
        # print(line)
        guid = i # (set_type, i)
        text_a = convert_to_unicode(line[1])
        encoded = tz.encode_plus(
            text=text_a,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=sequence_length,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_tensors='tf',  # ask the function to return PyTorch tensors
            return_attention_mask = False  # Generate the attention mask
            )
        input_id =  encoded['input_ids'];

        # text_a = tz.convert_tokens_to_ids(tz.tokenize(text_a))
        label = "0"
        guid_arr.append(guid)
        text_arr.append(input_id)

     input_mask = tf.random.uniform(
         shape=(1, sequence_length),
         maxval=vocab_size,
         dtype=tf.int32
     )

     ds_mask = tf.data.Dataset.from_tensor_slices(input_mask)
     ds_typeids = tf.data.Dataset.from_tensor_slices(guid_arr)
     ds_wordids = tf.data.Dataset.from_tensor_slices(text_arr)
     dataset = tf.data.Dataset.zip((ds_mask, ds_typeids, ds_wordids))

     dataset = dataset.repeat()
     dataset = dataset.batch(batch_size)
     dataset = dataset.take(count=1)  # loop over 1 batch
     dataset = dataset.cache()
     dataset = dataset.repeat()
     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

     return dataset

dataset = get_dataset2()
ds_iter = iter(dataset)
import pprint
for batch in ds_iter:
    pprint.pprint(batch)
    break
