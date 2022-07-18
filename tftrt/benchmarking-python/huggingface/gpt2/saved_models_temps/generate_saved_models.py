import argparse
import os
import shutil

from pprint import pprint

import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="gpt2 SavedModel Converter")

    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Directory containing where to export the saved model."
    )

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    model = TFGPT2Model.from_pretrained('gpt2')

    @tf.function
    def infer_fn(inputs):
        return model(inputs, use_cache=False)["last_hidden_state"]

    infer_fn_concrete = infer_fn.get_concrete_function({
        "input_ids": tf.TensorSpec([None, model.config.n_positions], dtype=tf.int32),
        "attention_mask": tf.TensorSpec([None, model.config.n_positions], dtype=tf.int32),
        "token_type_ids": tf.TensorSpec([None, model.config.n_positions], dtype=tf.int32),
    })

    # batch_size = 32
    # inputs = {
    #     "input_ids": tf.random.uniform([batch_size, model.config.n_positions], dtype=tf.int32, maxval=model.config.vocab_size),
    #     "attention_mask": tf.ones([batch_size, model.config.n_positions], dtype=tf.int32),
    #     "token_type_ids": tf.random.uniform([batch_size, model.config.n_positions], dtype=tf.int32, maxval=model.config.vocab_size),
    # }
    # data = infer_fn(inputs)

    try:
        shutil.rmtree(args.output_directory)
    except OSError: pass

    model_dir = os.path.join(args.output_directory, "model")
    tokenizer_dir = os.path.join(args.output_directory, "tokenizer")

    print(f"Saving `gpt2 model` in directory: `{model_dir}`.")

    tf.keras.models.save_model(
        model=model,
        filepath=model_dir,
        save_format="tf",
        overwrite=True,
        include_optimizer=False,
        signatures={"serving_default": infer_fn_concrete},
        options=None,
        save_traces=False,
    )

    print(f"Saving `gpt2 tokenizer` in directory: `{tokenizer_dir}`.")

    # Reload with `AutoTokenizer.from_pretrained(tokenizer_dir)`
    data = tokenizer.save_pretrained(tokenizer_dir)
    print("Tokenizer Data Saved:")
    pprint(data)
