import os
import copy
import argparse
import time

from statistics import mean

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

SAVEDMODEL_PATH = "./checkpoints/saved_model"

def load_with_converter(path, precision, batch_size):
    """Loads a saved model using a TF-TRT converter, and returns the converter
    """

    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
    if precision == 'int8':
        precision_mode = trt.TrtPrecisionMode.INT8
    elif precision == 'fp16':
        precision_mode = trt.TrtPrecisionMode.FP16
    else:
        precision_mode = trt.TrtPrecisionMode.FP32

    params = params._replace(
        precision_mode=precision_mode,
        max_workspace_size_bytes=2 << 32,  # 8,589,934,592 bytes
        maximum_cached_engines=100,
        minimum_segment_size=3,
        allow_build_at_runtime=True
    )

    import pprint
    print("%" * 85)
    pprint.pprint(params)
    print("%" * 85)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path,
        conversion_params=params,
    )

    return converter


if __name__ == "__main__":

    INFERENCE_STEPS = 10000
    WARMUP_STEPS = 2000

    parser = argparse.ArgumentParser()

    feature_parser = parser.add_mutually_exclusive_group(required=True)

    feature_parser.add_argument('--use_native_tensorflow', dest="use_tftrt", help="help", action='store_false')
    feature_parser.add_argument('--use_tftrt_model', dest="use_tftrt", action='store_true')

    parser.add_argument('--precision', dest="precision", type=str, default="fp16", choices=['int8', 'fp16', 'fp32'], help='Precision')
    parser.add_argument('--batch_size', dest="batch_size", type=int, default=512, help='Batch size')

    args = parser.parse_args()

    print("\n=========================================")
    print("Inference using: {} ...".format(
        "TF-TRT" if args.use_tftrt else "Native TensorFlow")
    )
    print("Batch size:", args.batch_size)
    if args.use_tftrt:
        print("Precision: ", args.precision)
    print("=========================================\n")
    time.sleep(2)

    def dataloader_fn(data_dir, batch_size):

        import tensorflow_datasets as tfds
        from official.vision.image_classification.mnist_main import decode_image

        mnist = tfds.builder('mnist', data_dir=data_dir)
        mnist.download_and_prepare()

        _, mnist_test = mnist.as_dataset(
            split=['train', 'test'],
            decoders={'image': decode_image()},
            as_supervised=True
        )

        ds = mnist_test.cache().repeat().batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    if args.use_tftrt:
        converter = load_with_converter(
            os.path.join(SAVEDMODEL_PATH),
            precision=args.precision,
            batch_size=args.batch_size
        )
        if args.precision == 'int8':
            num_calibration_batches = 2
            calibration_data = dataloader_fn('/data', args.batch_size)
            calibration_data = calibration_data.take(num_calibration_batches)
            def calibration_input_fn():
                for x, y in calibration_data:
                    yield (x, )
            xx = converter.convert(calibration_input_fn=calibration_input_fn)
        else:
            # fp16 or fp32
            xx = converter.convert()

        converter.save(
            os.path.join(SAVEDMODEL_PATH, "converted")
        )

        root = tf.saved_model.load(os.path.join(SAVEDMODEL_PATH, "converted"))
    else:
        root = tf.saved_model.load(SAVEDMODEL_PATH)

    infer = root.signatures['serving_default']
    output_tensorname = list(infer.structured_outputs.keys())[0]

    ds = dataloader_fn(
        data_dir="/data",
        batch_size=args.batch_size
    )
    iterator = iter(ds)
    features, labels = iterator.get_next()

    try:
        step_times = list()
        for step in range(1, INFERENCE_STEPS + 1):
            if step % 100 == 0:
                print("Processing step: %04d ..." % step)
            start_t = time.perf_counter()
            probs = infer(features)[output_tensorname]
            inferred_class = tf.math.argmax(probs).numpy()
            step_time = time.perf_counter() - start_t
            if step >= WARMUP_STEPS:
                step_times.append(step_time)
    except tf.errors.OutOfRangeError:
        pass

    avg_step_time = mean(step_times)
    print("\nAverage step time: %.1f msec" % (avg_step_time * 1e3))
    print("Average throughput: %d samples/sec" % (
        args.batch_size / avg_step_time
    ))
