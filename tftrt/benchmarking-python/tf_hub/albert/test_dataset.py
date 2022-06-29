import tensorflow as tf
import csv
import tensorflow_datasets as tfds

from dataloading import get_dataset_cola

def get_dataset(sequence_length=128, batch_size=32, vocab_size=32000):
    if False:
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
    else:
        return get_dataset_cola(sequence_length=sequence_length, batch_size=batch_size,
                vocab_size=vocab_size, full_path_to_file='test_cola.tsv')


dataset = get_dataset()
ds_iter = iter(dataset)
import pprint
for batch in ds_iter:
    pprint.pprint(batch)
    break
