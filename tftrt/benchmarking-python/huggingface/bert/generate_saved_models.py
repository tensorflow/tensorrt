#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

from transformers import BertTokenizer, TFBertModel
from transformers import BartTokenizer, TFBartModel

USE_CACHE = False
OUTPUT_ATTENTIONS = False
OUTPUT_HIDDEN_STATES = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="gpt2 SavedModel Converter")

    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Directory containing where to export the saved model."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Directory containing where to export the saved model."
    )

    args = parser.parse_args()

    # use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
    #     If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
    #     decoding (see :obj:`past_key_values`). Set to :obj:`False` during training, :obj:`True` during generation
    # output_attentions (:obj:`bool`, `optional`):
    #     Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
    #     tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
    #     config will be used instead.
    # output_hidden_states (:obj:`bool`, `optional`):
    #     Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
    #     more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
    #     used instead.
    model_kwargs = {
        "use_cache": USE_CACHE,
        "output_attentions": OUTPUT_ATTENTIONS,
        "output_hidden_states": OUTPUT_HIDDEN_STATES,
    }

    cache_dir = args.output_directory
    model_cache_dir = os.path.join(cache_dir, "keras_model")
    token_cache_dir = os.path.join(cache_dir, "tokenizer")
    pb_model_dir = os.path.join(cache_dir, "pb_model")

    try:
        shutil.rmtree(pb_model_dir)
    except FileNotFoundError:
        pass

    print()
    ("===============================================================")
    print("Processing model:", args.model_name)
    print("Will be saved to:", cache_dir, "\n")
    time.sleep(2)

    if args.model_name.startswith("bert"):
        if "uncased" in args.model_name:
            vocab_size = 30522
        else:
            vocab_size = 28996

        model = TFBertModel.from_pretrained(
            args.model_name, cache_dir=model_cache_dir, **model_kwargs
        )
        tokenizer = BertTokenizer.from_pretrained(
            args.model_name, cache_dir=token_cache_dir
        )

    else:
        vocab_size = 50265

        model = TFBartModel.from_pretrained(
            args.model_name, cache_dir=model_cache_dir, **model_kwargs
        )
        tokenizer = BartTokenizer.from_pretrained(
            args.model_name, cache_dir=token_cache_dir
        )

    print("Exporting Model to SavedModel at:", pb_model_dir)

    @tf.function(
        input_signature=[
            tf.TensorSpec((None, None), tf.int32, name="input_ids"),
            tf.TensorSpec((None, None), tf.int32, name="attention_mask")
        ]
    )
    def infer_func(input_ids, attention_mask):
        return model(
            input_ids=input_ids, attention_mask=attention_mask, training=False
        )

    tf.keras.models.save_model(
        model=model,
        filepath=pb_model_dir,
        overwrite=True,
        include_optimizer=False,
        save_format="tf",
        signatures={"serving_default": infer_func},
        options=None,
        save_traces=False
    )
    print("Done exporting to SavedModel ...")

    # Clearing the GPU memory

    del model
    del tokenizer
    K.clear_session()

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print()
