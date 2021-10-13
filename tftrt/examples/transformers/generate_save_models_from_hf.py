#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

from transformers import BertTokenizer, TFBertForPreTraining
from transformers import BartTokenizer, TFBartForConditionalGeneration


USE_CACHE = False
OUTPUT_ATTENTIONS = False
OUTPUT_HIDDEN_STATES = False


class HFModel(tf.Module):
    def __init__(self, model, is_encoder_decoder):
        self._model = model
        self._is_encoder_decoder = is_encoder_decoder

    @tf.function(
        input_signature=[tf.TensorSpec((None, None), tf.int32, name="input_ids")]
    )
    def serving(self, input_ids):
        if self._is_encoder_decoder:
            return self._model(
                input_ids, decoder_input_ids=input_ids, training=False
            )
        else:
            return self._model(input_ids, training=False)


if __name__ == "__main__":

    MODEL_NAMES = [
        "bert-base-uncased",
        "bert-base-cased",
        "bert-large-uncased",
        "bert-large-cased",
        'facebook/bart-base',
        'facebook/bart-large'
    ]


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

    for model_name in MODEL_NAMES:

        cache_dir = os.path.join("/models/", model_name)
        cache_dir = cache_dir.replace("-", "_")
        cache_dir = cache_dir.replace("facebook/", "")
        model_cache_dir = os.path.join(cache_dir, "keras_model")
        token_cache_dir = os.path.join(cache_dir, "tokenizer")
        pb_model_dir = os.path.join(cache_dir, "pb_model")

        try:
            shutil.rmtree(pb_model_dir)
        except FileNotFoundError:
            pass

        print()
        print("===============================================================")
        print("Processing model:", model_name)
        print("Will be saved to:", cache_dir, "\n")
        time.sleep(2)

        if model_name.startswith("bert"):
            is_encoder_decoder = False
            if "uncased" in model_name:
                vocab_size = 30522
            else:
                vocab_size = 28996

            model = TFBertForPreTraining.from_pretrained(
                model_name, cache_dir=model_cache_dir, **model_kwargs
            )
            tokenizer = BertTokenizer.from_pretrained(
                model_name, cache_dir=token_cache_dir
            )

        else:
            is_encoder_decoder = True
            vocab_size = 50265

            model = TFBartForConditionalGeneration.from_pretrained(
                model_name, cache_dir=model_cache_dir, **model_kwargs
            )
            tokenizer = BartTokenizer.from_pretrained(
                model_name, cache_dir=token_cache_dir
            )

        print("Exporting Model to SavedModel at:", pb_model_dir)
        hf_model = HFModel(model, is_encoder_decoder)  # necessary to define a custom input signature

        tf.saved_model.save(
            hf_model,
            pb_model_dir,
            signatures={"serving_default": hf_model.serving}
        )
        print("Done exporting to SavedModel ...")

        # Generating some inference artifacts to compare with TF-TRT results

        numpy_asset_dir = os.path.join(pb_model_dir, "numpy_assets")
        try:
            os.makedirs(numpy_asset_dir)
        except FileExistsError:
            pass

        np.random.seed(1234)
        input_data = np.random.randint(low=0, high=vocab_size, size=(32, 128))

        np_arrays_to_save = {
            "input_data": input_data,
        }

        output = hf_model.serving(input_data)

        for key in output.keys():
            np_arrays_to_save[key] = output[key].numpy()

        for key, val in np_arrays_to_save.items():
            print("saving:", key, "...")
            arr_save_path = os.path.join(numpy_asset_dir, '%s.npy' % key)
            np.save(arr_save_path, val)
            assert(np.allclose(np.load(arr_save_path), val))

        # Clearing the GPU memory

        del model
        del tokenizer
        K.clear_session()

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print()
