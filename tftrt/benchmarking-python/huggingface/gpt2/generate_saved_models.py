import argparse
import os
import shutil

from pprint import pprint

import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model, GPT2Config

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

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    config = GPT2Config.from_pretrained(args.model_name, use_cache=False)
    hf_model = TFGPT2Model.from_pretrained(args.model_name, use_cache=False)

    # Need a custom tf.Module to cleanup the exported SavedModel's signatures
    class Model(tf.Module):

        def __init__(self, model):
            self._model = model

        def __call__(self, *args, **kwargs):
            return self._model.transformer(*args, **kwargs)

        @property
        def config(self):
            return self._model.config

    model = Model(hf_model)

    infer_fn_concrete = tf.function(model.__call__).get_concrete_function({
        "input_ids":
            tf.TensorSpec([None, model.config.n_positions],
                          dtype=tf.int32,
                          name="input_ids"),
        "attention_mask":
            tf.TensorSpec([None, model.config.n_positions],
                          dtype=tf.int32,
                          name="attention_mask"),
        "token_type_ids":
            tf.TensorSpec([None, model.config.n_positions],
                          dtype=tf.int32,
                          name="token_type_ids"),
    })

    try:
        shutil.rmtree(args.output_directory)
    except OSError:
        pass

    model_dir = os.path.join(args.output_directory, "model")
    tokenizer_dir = os.path.join(args.output_directory, "tokenizer")

    print(f"Saving `gpt2 model` in directory: `{model_dir}`.")

    tf.saved_model.save(
        model,
        export_dir=model_dir,
        signatures={"serving_default": infer_fn_concrete},
    )

    print(
        f"Saving `{args.model_name} tokenizer` in directory: `{tokenizer_dir}`."
    )

    # Reload with `AutoTokenizer.from_pretrained(tokenizer_dir)`
    data = tokenizer.save_pretrained(tokenizer_dir)
    print("Tokenizer Data Saved:")
    pprint(data)
