import argparse, os, time, sys, glob, shutil, subprocess, pprint
import tensorflow as tf
import numpy as np
import preprocessing
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants

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

def get_dataset(model, data_files, batch_size, use_synthetic, mode='validation'):
    if use_synthetic:
        input_size = get_input_size(model)
        features = np.random.normal(
            loc=112, scale=70,
            size=(batch_size, input_size, input_size, 3)).astype(np.float32)
        features = np.clip(features, 0.0, 255.0)
        features = tf.convert_to_tensor(value=tf.compat.v1.get_variable(
            "features", dtype=tf.float32, initializer=tf.constant(features)))
        dataset = tf.data.Dataset.from_tensor_slices([features])
        dataset = dataset.repeat()
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
        else:
            raise ValueError("Mode must be either 'validation' or 'benchmark'")
    return dataset

def calibrate(graph_func,
              model,
              calib_files,
              num_calib_inputs,
              batch_size):
    dataset = get_dataset(model, calib_files, batch_size, False, 'validation')
    num_iterations = num_calib_inputs//batch_size
    for i, (batch_feats, _) in enumerate(dataset):
        if i > num_iterations:
            break
        start_time = time.time()
        batch_preds = graph_func(batch_feats)
        print("Calibration Iteration {}/{}".format(i, num_iterations))
        i += 1
     
def func_from_saved_model(saved_model_dir):
    loaded = tf.saved_model.load(saved_model_dir)
    infer = loaded.signatures["serving_default"]
    def wrap_func(*args, **kwargs):
        return infer(*args, **kwargs)['logits']
    return wrap_func
    

def get_frozen_func(model,
                    conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS,
                    model_dir=None,
                    use_trt=False,
                    engine_dir=None,
                    use_dynamic_op=False,
                    calib_files=None,
                    num_calib_inputs=None,
                    use_synthetic=False,
                    batch_size=None,
                    saved_model_dir=None,
                    root_saved_model_dir='./saved_models'):
    """Retreives a frozen SavedModel and applies TF-TRT
    model: str, the model name
    use_trt: bool, if true use TensorRT
    precision: str, floating point precision (FP32, FP16, or INT8)
    batch_size: int, batch size for TensorRT optimizations
    returns: tensorflow.SavedModel, the TensorRT compatible frozen graph.
    """
    saved_model_dir = saved_model_dir or os.path.join(root_saved_model_dir, model)
    cache = engine_dir is not None
    num_nodes = {}
    times = {}
    graph_sizes ={}

    if cache and use_trt:
        trt_saved_model_dir = "{}/saved_model_{}_{}_{}_{}/".format(
            engine_dir, model, int(use_trt), conversion_params.precision_mode,
            conversion_params.max_batch_size)
        if os.path.isdir(trt_saved_model_dir):
            print("Loading SavedModel from {}".format(trt_saved_model_dir))
            func = func_from_saved_model(trt_saved_model_dir)
            return func, num_nodes, times, graph_sizes
    
    if use_trt:
        loaded = tf.saved_model.load(saved_model_dir)
        infer = loaded.signatures['serving_default']
        ops = infer.graph.get_operations()
        num_nodes['native_tf'] = len(ops)
        del loaded
        del infer
        del ops
        start_time = time.time()
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=saved_model_dir,
            conversion_params=conversion_params,
        )
        converted_func = converter.convert()
        converted_graph_def = converter._converted_graph_def
        times['trt_conversion'] = time.time() - start_time
        num_nodes['tftrt_total'] = len(converted_graph_def.node)
        num_nodes['trt_only'] = len([1 for n in converted_graph_def.node if str(n.op)=='TRTEngineOp'])
        graph_sizes['trt'] = len(converted_graph_def.SerializeToString())/(1<<20)
        def wrap_func(*args, **kwargs):
            return converted_func(*args, **kwargs)['logits']

        if conversion_params.precision_mode == 'INT8':
            print('Calibrating INT8')
            calibrate(converted_func, model, calib_files, num_calib_inputs,
                      batch_size)
            print('Done calibrating INT8')
            calib_dir = 'calibrated/{}'.format(model)
            converter.save(calib_dir)
            calibrated_func = func_from_saved_model(calib_dir)
            return calibrated_func, num_nodes, times, graph_sizes
        else:
            if cache:
                converter.save(trt_saved_model_dir)
            return wrap_func, num_nodes, times, graph_sizes
    else:
        func = func_from_saved_model(saved_model_dir)
        return func, num_nodes, times, graph_sizes

    if cache and use_trt:
        if not os.path.exists(os.path.dirname(trt_saved_model_dir)):
            try:
                os.makedirs(os.path.dirname(trt_saved_model_dir))
            except Exception as e:
                raise e
        start_time = time.time()
        converter.save(trt_saved_model_dir)
        times['saving_frozen_graph'] = time.time() - start_time

    return converted_func, num_nodes, times, graph_sizes

def eval_fn(model, preds, labels, adjust):
    preds = np.argmax(preds.numpy(), axis=1).reshape(-1) - adjust
    return np.sum((np.array(labels).reshape(-1) == preds).astype(np.float32))

def run(graph_func, model, data_files, batch_size,
    num_iterations, num_warmup_iterations, use_synthetic, display_every=100,
    mode='validation', target_duration=None):
    '''Run the given graph_func on the data files provided. In validation mode,
    it consumes TFRecords with labels and reports accuracy. In benchmark mode, it
    times inference on real data (.jpgs).'''
    results = {}
    corrects = 0
    iter_times = []
    adjust = 1 if get_num_classes(model) == 1001 else 0
    initial_time = time.time()
    dataset = get_dataset(model, data_files, batch_size, use_synthetic, mode)
    if mode == 'validation':
        for i, (batch_feats, batch_labels) in enumerate(dataset):
            start_time = time.time()
            batch_preds = graph_func(batch_feats)
            iter_times.append(time.time() - start_time)
            if i % display_every == 1:
                print("Iteration {}/{}".format(i - 1, 50000//batch_size))
            corrects += eval_fn(model, batch_preds, batch_labels, adjust)
            if i > 1 and target_duration is not None and \
                time.time() - initial_time > target_duration:
                break
        accuracy = corrects / (batch_size * i)
        results['accuracy'] = accuracy

    elif mode == 'benchmark':
        for i, batch_feats in enumerate(dataset):
            if i > num_warmup_iterations:
                start_time = time.time()
                outs = graph_func(batch_feats)
                iter_times.append(time.time() - start_time)
            else:
                outs = graph_func(batch_feats)
            if i % display_every == 1:
                print("Iteration {}/{}".format(i-1, num_iterations))
            if i > 1 and target_duration is not None and \
                time.time() - initial_time > target_duration:
                break
            if num_iterations is not None and i > num_iterations:
                break

    if not iter_times:
        return results
    iter_times = np.array(iter_times)
    iter_times = iter_times[num_warmup_iterations:]
    results['total_time'] = np.sum(iter_times)
    results['images_per_sec'] = np.mean(batch_size / iter_times)
    results['99th_percentile'] = np.percentile(iter_times, q=99, interpolation='lower') * 1000
    results['latency_mean'] = np.mean(iter_times) * 1000
    results['latency_median'] = np.median(iter_times) * 1000
    results['latency_min'] = np.min(iter_times) * 1000
    return results
    
    

def get_trt_conversion_params(max_workspace_size_bytes,
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
    return conversion_params

def setup_gpu_mem(gpu_mem_cap):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if not gpu_mem_cap:
                    tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem_cap)])

            print(len(gpus), "Physical GPU's")
        except RuntimeError as e:
            print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, default='resnet_v1_50',
        choices=list(NETS.keys()),
        help='Which model to use.')
    parser.add_argument('--data_dir', type=str, default=None,
        help='Directory containing validation set TFRecord files.')
    parser.add_argument('--calib_data_dir', type=str,
        help='Directory containing TFRecord files for calibrating INT8.')
    parser.add_argument('--root_saved_model_dir', type=str, default=None,
        help='Directory containing saved models.')
    parser.add_argument('--saved_model_dir', type=str, default=None,
        help='Directory containing a particular saved model.')
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
    parser.add_argument('--num_iterations', type=int, default=2048,
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
    parser.add_argument('--max_workspace_size', type=int, default=(1<<30),
        help='workspace size in bytes')
    parser.add_argument('--mode', choices=['validation', 'benchmark'], default='validation',
        help='Which mode to use (validation or benchmark)')
    parser.add_argument('--target_duration', type=int, default=None,
        help='If set, script will run for specified number of seconds.')
    parser.add_argument('--gpu_mem_cap', type=int, default=0,
        help='Set the maximum GPU memory cap in MB. If 0, allow growth will be used.')
    args = parser.parse_args()

    if args.precision == 'INT8':
        raise ValueError('INT8 is broken at this point, and is undergoing API '
        'changes. Please use 1.14 scripts for INT8 inference.')
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
    if args.root_saved_model_dir is None and args.saved_model_dir is None:
        raise ValueError("Please set one of --root_saved_model_dir or --saved_model_dir")
    if args.root_saved_model_dir is not None and args.saved_model_dir is not None:
        print("Both --root_saved_model_dir and --saved_model_dir are set.\n \
               Using saved_model_dir:{}".format(args.saved_model_dir))

    setup_gpu_mem(args.gpu_mem_cap)

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
        saved_model_dir=args.saved_model_dir,
        root_saved_model_dir=args.root_saved_model_dir)

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
    print('Results of {}'.format(args.model))
    def print_dict(input_dict, str=''):
        for k, v in sorted(input_dict.items()):
            headline = '{}({}): '.format(str, k) if str else '{}: '.format(k)
            print('{}{}'.format(headline, '%.1f'%v if type(v)==float else v))
    print_dict(vars(args)) 
    print_dict(num_nodes, str='num_nodes')
    if args.mode == 'validation':
        print('    accuracy: %.2f' % (results['accuracy'] * 100))
    print('    images/sec: %d' % results['images_per_sec'])
    print('    99th_percentile(ms): %.2f' % results['99th_percentile'])
    print('    total_time(s): %.1f' % results['total_time'])
    print('    latency_mean(ms): %.2f' % results['latency_mean'])
    print('    latency_median(ms): %.2f' % results['latency_median'])
    print('    latency_min(ms): %.2f' % results['latency_min']) 
