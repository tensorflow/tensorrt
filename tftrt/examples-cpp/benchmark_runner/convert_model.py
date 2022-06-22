# A simple TF2 script that converts a SavedModel with TRT at half-precision and saves it to disk.

import argparse
import copy
import os
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def parse_args():
    parser = argparse.ArgumentParser(
        description="TF-TRT Model Conversion"
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        type=str,
        default=None,
        help="Saved model directory.",
        required=True
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Location to save the converted model.",
        required=True
    )
    parser.add_argument(
        "--use-fp16",
        dest="fp16",
        action="store_true"
    )
    return parser.parse_args()

def load_model(path):
    return tf.saved_model.load(
        path,
        tags=[tag_constants.SERVING]
    ).signatures['serving_default']

def main(args):
    model = load_model(args.model_dir)

    # Convert with TRT
    precision_mode = trt.TrtPrecisionMode.FP32
    if args.fp16:
        precision_mode = trt.TrtPrecisionMode.FP16
    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
    params = params._replace(
        precision_mode=precision_mode,
        allow_build_at_runtime=True
    )
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=args.model_dir,
        conversion_params=params,
    )
    converter.convert()

    os.makedirs(args.output_dir, exist_ok=True)
    converter.save(args.output_dir)
    model = load_model(args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
