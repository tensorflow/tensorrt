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
        signatures=None,
        options=None,
        save_traces=True,
    )

    print(f"Saving `gpt2 tokenizer` in directory: `{tokenizer_dir}`.")

    # Reload with `AutoTokenizer.from_pretrained(tokenizer_dir)`
    data = tokenizer.save_pretrained(tokenizer_dir)
    print("Tokenizer Data Saved:")
    pprint(data)

