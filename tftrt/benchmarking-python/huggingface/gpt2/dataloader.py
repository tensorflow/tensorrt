############## Require T5 version 0.4.0 #############

import os
import glob

try:
    from prefetch_generator import background
except ModuleNotFoundError:
    print("[ERROR] Please install: `pip install --upgrade prefetch_generator`")
    raise

try:
    import orjson as json
except ModuleNotFoundError:
    print(
        "[WARNING] To process json data faster, please execute: "
        "`pip install --upgrade orjson`"
    )
    import json

import numpy as np
import tensorflow as tf

import t5.data.preprocessors as prep
from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary
from transformers import T5Tokenizer


def get_dataset_c4(
    data_dir,
    vocab_dir,
    tokenizer_dir=None,
    sequence_length=128,
    batch_size=32,
    vocab_size=512,
    noise_density=0.15
):
    json_files = sorted(glob.glob(
        os.path.join(data_dir, "c4-validation.*.json")
    ))

    if tokenizer_dir is None:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
    else:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)

    @background(max_prefetch=1)
    def jsonfile_parser(filename):

        for line in open(filename):
            data = json.loads(line)

            yield {
                "targets": np.squeeze(tokenizer(
                    data["text"],
                    return_tensors="tf",
                    max_length=sequence_length,
                    truncation=True,
                    padding="max_length",
                ).input_ids)
            }

    def _get_ds_generator(_filename):
       return tf.data.Dataset.from_generator(
            lambda: jsonfile_parser(_filename),
            output_signature={
                "targets": tf.TensorSpec(
                    shape=(sequence_length,),
                    dtype=tf.int32,
                    name=None
                )
            },
        ).prefetch(buffer_size=tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.sample_from_datasets(
        datasets=[_get_ds_generator(_f) for _f in json_files],
        seed=666,
        stop_on_empty_dataset=False
    )

    vocabulary = SentencePieceVocabulary(
        sentencepiece_model_file=os.path.join(vocab_dir, "spiece.model"),
        extra_ids=0
    )
    dataset = prep.denoise(
            dataset,
            vocabulary,
            noise_density=noise_density,
            noise_mask_fn=prep.random_spans_noise_mask,
            inputs_fn=prep.noise_token_to_sentinel,
            targets_fn=None
        )
    
    def transform_fn(features):
        pad_token_id = tokenizer.pad_token_id

        # Decoder token set to pad token by default.
        decoder_start_token_id = pad_token_id
    
        # Shift labels to right by one to create decoder inputs.
        decoder_input_ids = tf.concat(
            [[decoder_start_token_id],
            features["targets"][:-1]],
            axis = 0
        )

        # Change -100 to pad token to prevent ignorance.
        decoder_input_ids = tf.where(
            tf.equal(decoder_input_ids, -100),
            tf.fill(decoder_input_ids.shape.as_list(), pad_token_id),
            decoder_input_ids
        )

        # Set All Attention Masks to 1 when no padding on inputs given.
        return {
                "attention_mask": tf.ones_like(features["inputs"]),
                "decoder_attention_mask": tf.ones_like(decoder_input_ids),
                "decoder_input_ids": decoder_input_ids,
                "input_ids": features["inputs"],
                "targets": features["targets"]
            }
    
    dataset = dataset.map(
        transform_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch an entire batch of data before batching
    dataset = dataset.prefetch(buffer_size=batch_size)

    # Then Batch
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset
