import argparse
import os
import shutil

from pprint import pprint

import tensorflow as tf

from transformers import AutoTokenizer
from transformers import TFAutoModelWithLMHead
from transformers import TFAutoModelForSeq2SeqLM

SUPPORTED_MODELS = ["t5-small", "t5-base", "t5-large", "google/t5-v1_1-base"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="T5 SavedModel Converter")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="The name of the model to convert to SavedModel."
    )

    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Directory containing where to export the saved model."
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if "google" in args.model_name:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    else:
        model = TFAutoModelWithLMHead.from_pretrained(args.model_name)

    try:
        shutil.rmtree(args.output_directory)
    except OSError:
        pass

    model_dir = os.path.join(args.output_directory, "model")
    tokenizer_dir = os.path.join(args.output_directory, "tokenizer")

    print(
        f"\n[INFO] Saving `{args.model_name} model` in directory: `{model_dir}`."
    )

    tf.keras.models.save_model(
        model=model,
        filepath=model_dir,
        overwrite=True,
        include_optimizer=False,
        save_format="tf",
        signatures=model.serving,
        options=None,
        save_traces=True
    )

    print(
        f"\n[INFO] Saving `{args.model_name} tokenizer` in directory: `{tokenizer_dir}`."
    )

    # Reload with `AutoTokenizer.from_pretrained(tokenizer_dir)`
    data = tokenizer.save_pretrained(tokenizer_dir)
    print("[INFO] Tokenizer Data Saved:")
    pprint(data)
