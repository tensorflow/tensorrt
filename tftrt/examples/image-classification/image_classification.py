import argparse, os, time, sys, glob, shutil, subprocess, pprint
import tensorflow as tf
import numpy as np
import preprocessing
from tensorflow.python.compiler.tensorrt import trt_convert as trt

NETS = {
    'mobilenet_v1': {
        'preprocess':'inception',
        'input_size':224,
        'num_classes':1001},
    'mobilenet_v2': {
        'preprocess':'inception',
        'input_size':224,
        'num_classes':1001},
    'nasnet_mobile': {
        'preprocess':'inception',
        'input_size':224,
        'num_classes':1001},
    'nasnet_large': {
        'preprocess':'inception',
        'input_size':331,
        'num_classes':1001},
    'resnet_v1_50': {
        'preprocess':'vgg',
        'input_size':224,
        'num_classes':1001},
    'resnet_v2_50': {
        'preprocess':'vgg',
        'input_size':224,
        'num_classes':1001},
    'resnet_v2_152': {
        'preprocess':'vgg',
        'input_size':224,
        'num_classes':1001},
    'vgg_16': {
        'preprocess':'vgg',
        'input_size':224,
        'num_classes':1000},
    'vgg_19': {
        'preprocess':'vgg',
        'input_size':224,
        'num_classes':1000},
    'inception_v3': {
        'preprocess':'inception',
        'input_size':299,
        'num_classes':1001},
    'inception_v4': {
        'preprocess':'inception',
        'input_size':299,
        'num_classes':1001},
}

def get_input_size(model):
    return NETS[model]['input_size']

def get_preprocessing(model):
    if NETS[model]['preprocess'] == 'vgg':
        return preprocessing.vgg_preprocess
    else:
        return preprocessing.inception_preprocess

def get_num_classes(model):
    return NETS[model]['num_classes']

def deserialize_image_record(record):
    feature_map = {
        'image/encoded':          tf.io.FixedLenFeature([ ], tf.string, ''),
        'image/class/label':      tf.io.FixedLenFeature([1], tf.int64,  -1),
        'image/class/text':       tf.io.FixedLenFeature([ ], tf.string, ''),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
    }
    with tf.compat.v1.name_scope('deserialize_image_record'):
        obj = tf.io.parse_single_example(serialized=record, features=feature_map)
        imgdata = obj['image/encoded']
        label   = tf.cast(obj['image/class/label'], tf.int32)
        return imgdata, label

def get_preprocess_fn(model, mode='validation'):
    """Creates a function to parse and process a TFRecord using the model's parameters

    model: string, the model name (see NETS table)
    mode: string, which mode to use (validation or benchmark)
    returns: function, the preprocessing function for a record
    """
    preprocess_fn = get_preprocessing(model)
    input_size = get_input_size(model)

    def validation_process(record):
        # Parse TFRecord
        imgdata, label = deserialize_image_record(record)
        label -= 1 # Change to 0-based (don't use background class)
        try:    image = tf.image.decode_jpeg(imgdata, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
        except: image = tf.image.decode_png(imgdata, channels=3)
        # Use model's preprocessing function
        image = preprocess_fn(image, input_size, input_size)
        return image, label

    def benchmark_process(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = preprocess_fn(image, input_size, input_size)
        return image

    if mode == 'validation':
        return validation_process
    elif mode == 'benchmark':
        return benchmark_process
    else:
        raise ValueError("Mode must be either 'validation' or 'benchmark'")

def input_fn(model, data_files, batch_size, use_synthetic, mode='validation'):
    if use_synthetic:
        input_size = get_input_size(model)
        features = np.random.normal(
            loc=112, scale=70,
            size=(batch_size, input_size, input_size, 3)).astype(np.float32)
        features = np.clip(features, 0.0, 255.0)
        features = tf.convert_to_tensor(value=tf.compat.v1.get_variable(
            "features", dtype=tf.float32, initializer=tf.constant(features)))
        labels = np.random.randint(
            low=0,
            high=get_num_classes(model),
            size=(batch_size),
            dtype=np.int32)
        labels = tf.identity(tf.constant(labels))
    else:
        # preprocess function for input data
        preprocess_fn = get_preprocess_fn(model, mode)
        num_classes = get_num_classes(model)
        if mode == 'validation':
            dataset = tf.data.TFRecordDataset(data_files)
            dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=8)
            dataset = dataset.batch(batch_size=batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            dataset = dataset.repeat(count=1)
        elif mode == 'benchmark':
            dataset = tf.data.Dataset.from_tensor_slices(data_files)
            dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=8)
            dataset = dataset.batch(batch_size=batch_size)
            dataset = dataset.repeat(count=1)
            labels = np.random.randint(
                low=0,
                high=num_classes,
                size=(batch_size),
                dtype=np.int32)
            labels = tf.identity(tf.constant(labels))
        else:
            raise ValueError("Mode must be either 'validation' or 'benchmark'")
    return dataset

def get_frozen_func(model,
                    conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS,
                    model_dir=None,
                    use_trt=False,
                    engine_dir=None,
                    use_dynamic_op=False,
                    calib_files=None,
                    num_calib_inputs=None,
                    use_synthetic=False,
                    cache=False,
                    batch_size=None,
                    default_models_dir='./saved_models'):
    """Retreives a frozen SavedModel and applies TF-TRT
    model: str, the model name
    use_trt: bool, if true use TensorRT
    precision: str, floating point precision (FP32, FP16, or INT8)
    batch_size: int, batch size for TensorRT optimizations
    returns: tensorflow.SavedModel, the TensorRT compatible frozen graph.
    """
    num_nodes = {}
    times = {}
    graph_sizes ={}
    print(params.precision_mode)
    
    saved_model_path = "converted/saved_model_%s_%d_%s_%d/" % (model, int(use_trt), conversion_params.precision_mode, conversion_params.max_batch_size)
    if cache and use_trt:
        if os.path.isdir(saved_model_path):
            print("---------loading from {}".format(saved_model_path))
            loaded = tf.saved_model.load(saved_model_path)

            infer = loaded.signatures["serving_default"]
            def wrap_func(*args, **kwargs):
                print(infer(*args, **kwargs).keys())
                return infer(*args, **kwargs)['output_0']
            return wrap_func, num_nodes, times, graph_sizes
        else:
            print("-----------Compiling new function")
    
    if use_trt:
        start_time = time.time()
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=os.path.join(default_models_dir, model+'VCCB'),
            input_saved_model_tags=None,
            input_saved_model_signature_key=None,
            conversion_params=conversion_params,
        )
        converted_func = converter.convert()
        converted_graph_def = converter._converted_graph_def
        times['trt_conversion'] = time.time() - start_time
        num_nodes['tftrt_total'] = len(converted_graph_def.node)
        num_nodes['trt_only'] = len([1 for n in converted_graph_def.node if str(n.op)=='TRTEngineOp'])
        graph_sizes['trt'] = len(converted_graph_def.SerializeToString())
        def wrap_func(*args, **kwargs):
            return converted_func(*args, **kwargs)['logits']
        if conversion_params.precision_mode == 'INT8':
            run(wrap_func, model, calib_files, batch_size, 10, 10, False, mode='validation') 
            converter.save(saved_model_path)
            loaded = tf.saved_model.load(saved_model_path)
            infer = loaded.signatures['serving_default']
            def wrap_func(*args, **kwargs):
                return infer(*args, **kwargs)['logits']
            return wrap_func, num_nodes, times, graph_sizes

        return wrap_func, num_nodes, times, graph_sizes
    else:
        loaded = tf.saved_model.load(os.path.join(default_models_dir, model+'VCCB'))
        infer = loaded.signatures['serving_default']
        def wrap_func(*args, **kwargs):
            return infer(*args, **kwargs)['logits']
        return wrap_func, num_nodes, times, graph_sizes

    #calibration

    if cache and use_trt:
        if not os.path.exists(os.path.dirname(saved_model_path)):
            try:
                os.makedirs(os.path.dirname(saved_model_path))
            except Exception as e:
                raise e
        start_time = time.time()
        converter.save(saved_model_path)
        times['saving_frozen_graph'] = time.time() - start_time

    return converted_func, num_nodes, times, graph_sizes

def eval_fn(model, preds, labels, adjust):
    preds = np.array(preds.numpy())
    preds = np.argmax(preds.reshape(-1))
    preds -= adjust
    return np.sum((np.array(labels).reshape(-1) == preds).astype(np.float32))

def run(graph_func, model, data_files, batch_size,
    num_iterations, num_warmup_iterations, use_synthetic, display_every=100,
    mode='validation', target_duration=None):

    results = {}
    print(batch_size)

    dataset = input_fn(model, data_files, batch_size, use_synthetic, mode)
    i = 0
    corrects = 0
    iter_times = []
    adjust = 1 if get_num_classes(model) == 1001 else 0
    print("Model {} adj is {}".format(model, adjust))

    if mode == 'validation':
        for batch_feats, batch_labels in dataset:
            start_time = time.time()
            batch_preds = graph_func(batch_feats)
            iter_times.append(time.time() - start_time)
            if i % 100 == 0:
                print("Iteration {}, Images processed {}".format(i, i * batch_size))
            corrects += eval_fn(model, batch_preds, batch_labels, adjust)
            i += 1
        accuracy = corrects / (batch_size * i)
        print("accuracy:")
        print(accuracy)

    elif mode == 'benchmark':
        for i, batch_feats in enumerate(dataset):
            outs = graph_func(batch_feats)
            if i == num_warmup_iterations:
                start_time = time.time()
            if i > num_warmup_iterations:
                iter_times.append(time.time() - start_time)
                start_time = time.time()
            if i % 100 == 0:
                print("Iteration {}, Images processed {}".format(i, i * batch_size))
            if num_iterations is not None and i > num_iterations:
                break
    
    iter_times = np.array(iter_times)
    print("---stats---")
    #iter_times = iter_times[num_warmup_iterations:]
    print(len(iter_times))
    results['total_time'] = np.sum(iter_times)
    results['images_per_sec'] = np.mean(batch_size / iter_times)
    results['99th_percentile'] = np.percentile(iter_times, q=99, interpolation='lower') * 1000
    results['latency_mean'] = np.mean(iter_times) * 1000
    results['latency_median'] = np.median(iter_times) * 1000
    results['latency_min'] = np.min(iter_times) * 1000
    return results
    
    

def get_trt_conversion_params(
    max_workspace_size_bytes,
    precision_mode,
    minimum_segment_size,
    is_dynamic_op,
    max_batch_size):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=max_workspace_size_bytes)
    conversion_params = conversion_params._replace(precision_mode=precision_mode)
    conversion_params = conversion_params._replace(minimum_segment_size=minimum_segment_size)
    conversion_params = conversion_params._replace(is_dynamic_op=is_dynamic_op)
    conversion_params = conversion_params._replace(use_calibration=precision_mode=='INT8')
    conversion_params = conversion_params._replace(max_batch_size=max_batch_size)
    print(conversion_params)
    return conversion_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, default='inception_v4',
        choices=['mobilenet_v1', 'mobilenet_v2', 'nasnet_mobile', 'nasnet_large',
                 'resnet_v1_50', 'resnet_v2_50', 'resnet_v2_152', 'vgg_16', 'vgg_19',
                 'inception_v3', 'inception_v4'],
        help='Which model to use.')
    parser.add_argument('--data_dir', type=str, default=None,
        help='Directory containing validation set TFRecord files.')
    parser.add_argument('--calib_data_dir', type=str,
        help='Directory containing TFRecord files for calibrating INT8.')
    parser.add_argument('--model_dir', type=str, default=None,
        help='Directory containing model checkpoint. If not provided, a ' \
             'checkpoint may be downloaded automatically and stored in ' \
             '"{--default_models_dir}/{--model}" for future use.')
    parser.add_argument('--default_models_dir', type=str, default='./data',
        help='Directory where downloaded model checkpoints will be stored and ' \
             'loaded from if --model_dir is not provided.')
    parser.add_argument('--use_trt', action='store_true',
        help='If set, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--engine_dir', type=str, default=None,
        help='Directory where to write trt engines. Engines are written only if the directory ' \
             'is provided. This option is ignored when not using tf_trt.')
    parser.add_argument('--use_trt_dynamic_op', action='store_true',
        help='If set, TRT conversion will be done using dynamic op instead of statically.')
    parser.add_argument('--precision', type=str, choices=['FP32', 'FP16', 'INT8'], default='FP32',
        help='Precision mode to use. FP16 and INT8 only work in conjunction with --use_trt')
    parser.add_argument('--batch_size', type=int, default=8,
        help='Number of images per batch.')
    parser.add_argument('--minimum_segment_size', type=int, default=2,
        help='Minimum number of TF ops in a TRT engine.')
    parser.add_argument('--num_iterations', type=int, default=None,
        help='How many iterations(batches) to evaluate. If not supplied, the whole set will be evaluated.')
    parser.add_argument('--display_every', type=int, default=100,
        help='Number of iterations executed between two consecutive display of metrics')
    parser.add_argument('--use_synthetic', action='store_true',
        help='If set, one batch of random data is generated and used at every iteration.')
    parser.add_argument('--num_warmup_iterations', type=int, default=50,
        help='Number of initial iterations skipped from timing')
    parser.add_argument('--num_calib_inputs', type=int, default=500,
        help='Number of inputs (e.g. images) used for calibration '
        '(last batch is skipped in case it is not full)')
    parser.add_argument('--max_workspace_size', type=int, default=(1<<32),
        help='workspace size in bytes')
    parser.add_argument('--cache', action='store_true',
        help='If set, graphs will be saved to disk after conversion. If a converted graph is present on disk, it will be loaded instead of building the graph again.')
    parser.add_argument('--mode', choices=['validation', 'benchmark'], default='validation',
        help='Which mode to use (validation or benchmark)')
    parser.add_argument('--target_duration', type=int, default=None,
        help='If set, script will run for specified number of seconds.')
    args = parser.parse_args()

    if args.precision != 'FP32' and not args.use_trt:
        raise ValueError('TensorRT must be enabled for FP16 or INT8 modes (--use_trt).')
    if args.precision == 'INT8' and not args.calib_data_dir and not args.use_synthetic:
        raise ValueError('--calib_data_dir is required for INT8 mode')
    if args.num_iterations is not None and args.num_iterations <= args.num_warmup_iterations:
        raise ValueError('--num_iterations must be larger than --num_warmup_iterations '
            '({} <= {})'.format(args.num_iterations, args.num_warmup_iterations))
    if args.num_calib_inputs < args.batch_size:
        raise ValueError('--num_calib_inputs must not be smaller than --batch_size'
            '({} <= {})'.format(args.num_calib_inputs, args.batch_size))
    if args.mode == 'validation' and args.use_synthetic:
        raise ValueError('Cannot use both validation mode and synthetic dataset')
    if args.data_dir is None and not args.use_synthetic:
        raise ValueError("--data_dir required if you are not using synthetic data")
    if args.use_synthetic and args.num_iterations is None:
        raise ValueError("--num_iterations is required for --use_synthetic")

    def get_files(data_dir, filename_pattern):
        if data_dir == None:
            return []
        files = tf.io.gfile.glob(os.path.join(data_dir, filename_pattern))
        if files == []:
            raise ValueError('Can not find any files in {} with '
                             'pattern "{}"'.format(data_dir, filename_pattern))
        return files

    calib_files = []
    data_files = []
    if not args.use_synthetic:
        if args.mode == "validation":
            data_files = get_files(args.data_dir, 'validation*')
        elif args.mode == "benchmark":
            data_files = [os.path.join(path, name) for path, _, files in os.walk(args.data_dir) for name in files]
        else:
            raise ValueError("Mode must be either 'validation' or 'benchamark'")
        calib_files = get_files(args.calib_data_dir, 'train*')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(len(gpus), "Physical GPU's")
        except RuntimeError as e:
            print(e)

    params = get_trt_conversion_params(
        args.max_workspace_size,
        args.precision,
        args.minimum_segment_size,
        args.use_trt_dynamic_op,
        args.batch_size,)

    graph_func, num_nodes, times, graph_sizes = get_frozen_func(
        model=args.model,
        conversion_params=params,
        use_trt=args.use_trt,
        engine_dir=args.engine_dir,
        use_dynamic_op=args.use_trt_dynamic_op,
        calib_files=calib_files,
        batch_size=args.batch_size,
        num_calib_inputs=args.num_calib_inputs,
        use_synthetic=args.use_synthetic,
        cache=args.cache,
        default_models_dir='./saved_models',)
    pprint.pprint(num_nodes)    
    pprint.pprint(times)    
    pprint.pprint(graph_sizes)

    results = run(
        graph_func,
        model=args.model,
        data_files=data_files,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        use_synthetic=args.use_synthetic,
        display_every=args.display_every,
        mode=args.mode,
        target_duration=args.target_duration)
    pprint.pprint(results)
