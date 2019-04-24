import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import time
import random
import numpy as np
import pandas as pd
from official.datasets import movielens

from neumf import compute_eval_metrics
from neumf import neural_mf
import os
import argparse
import csv

class LoggerHook(tf.train.SessionRunHook):
    """Logs runtime of each iteration"""
    def __init__(self, batch_size, num_records, display_every):
        self.iter_times = []
        self.display_every = display_every
        self.num_steps = (num_records + batch_size - 1) / batch_size
        self.batch_size = batch_size

    def before_run(self, run_context):
        self.start_time = time.time()

    def after_run(self, run_context, run_values):
        current_time = time.time()
        duration = current_time - self.start_time
        self.iter_times.append(duration)
        current_step = len(self.iter_times)
        if current_step % self.display_every == 0:
            print("    step %d/%d, iter_time(ms)=%.4f, images/sec=%d" % (
                current_step, self.num_steps, duration * 1000,
                self.batch_size / self.iter_times[-1]))

class BenchmarkHook(tf.train.SessionRunHook):
    """Limits run duration and number of iterations"""
    def __init__(self, target_duration=None, iteration_limit=None):
        self.target_duration = target_duration
        self.start_time = None
        self.current_iteration = 0
        self.iteration_limit = iteration_limit
    def before_run(self, run_context):
        if not self.start_time:
            self.start_time = time.time()
            if self.target_duration:
                print("    running for target duration {} seconds from {}".format(
                    self.target_duration, time.asctime(time.localtime(self.start_time))))

    def after_run(self, run_context, run_values):
        if self.target_duration:
            current_time = time.time()
            if (current_time - self.start_time) > self.target_duration:
                print("    target duration {} reached at {}, requesting stop".format(
                    self.target_duration), time.asctime(time.localtime(current_time)))
                run_context.request_stop()
        if self.iteration_limit:
            self.current_iteration += 1
            if self.current_iteration >= self.iteration_limit:
                run_context.request_stop()


def get_frozen_graph(model_checkpoint="/data/marek_ckpt/model.ckpt",
                     mode="benchmark",
                     use_trt=True,
                     batch_size=1024,
                     use_dynamic_op=True,
                     precision="FP32",
                     model_dtype=tf.float32,
                     mf_dim=64,
                     mf_reg=64,
                     mlp_layer_sizes=[256, 256, 128, 64],
                     mlp_layer_regs=[.0, .0, .0, .0],
                     nb_items=26744,
                     nb_users=138493,
                     dup_mask=0.1,
                     K=10,
                     minimum_segment_size=2,
                     calib_data_dir=None,
                     num_calib_inputs=None,
                     use_synthetic=False,
                     max_workspace_size=(1<<32)):

    num_nodes = {}
    times = {}
    graph_sizes = {}

    tf_config = tf.ConfigProto()
    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
            users = tf.placeholder(shape=(None,), dtype=tf.int32, name="user_input")
            items = tf.placeholder(shape=(None,), dtype=tf.int32, name="item_input")
            with tf.variable_scope("neumf"):
                logits = neural_mf(users, items, model_dtype, nb_users, nb_items, mf_dim, mf_reg, mlp_layer_sizes, mlp_layer_regs, 0.1)

            saver = tf.train.Saver()
            saver.restore(tf_sess, "/data/marek_ckpt/model.ckpt")
            graph0 = tf.graph_util.convert_variables_to_constants(tf_sess,
                tf_sess.graph_def, output_node_names=['neumf/dense_3/BiasAdd'])
            frozen_graph = tf.graph_util.remove_training_nodes(graph0)

            for node in frozen_graph.node:
                if node.op == "Assign":
                    node.op = "Identity"
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                    if 'validate_shape' in node.attr: del node.attr['validate_shape']
                    if len(node.input) == 2:
                        node.input[0] = node.input[1]
                        del node.input[1]

    if use_trt:
        start_time = time.time()
        frozen_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=['neumf/dense_3/BiasAdd:0'],
            max_batch_size=batch_size,
            max_workspace_size_bytes=max_workspace_size,
            precision_mode=precision_mode,
            minimum_segment_size=minimum_segment_size,
            is_dynamic_op=use_dynamic_op)
        times['trt_conversion'] = time.time() - start_time
        num_nodes['tftrt_total']=len(frozen_graph.node)
        num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp'])
        graph_sizes['trt'] = len(frozen_graph.SerializeToString())
        
        if precision == 'int8':
            calib_graph = frozen_graph
            graph_size['calib'] = len(calib_graph.SerializeToString())
            # INT8 calibration step
            print('Calibrating INT8...')
            start_time = time.time()
            run(calib_graph,
                data_dir=calib_data_dir,
                batch_size=batch_size,
                num_iterations=num_calib_inputs // batch_size, 
                num_warmup_iterations=0,
                use_synthetic=use_synthetic)
            times['trt_calibration'] = time.time() - start_time
            start_time = time.time()
            frozen_graph = trt.calib_graph_to_infer_graph(calib_graph)
            times['trt_int8_conversion'] = time.time() - start_time
            graph_sizes['trt'] = len(frozen_graph.SerializeToString())

            del calib_graph
            print('INT8 graph created')

    return frozen_graph, num_nodes, times, graph_sizes


def run(frozen_graph,
        data_dir='/data/ml-20m/',
        batch_size=1024,
        num_iterations=None,
        num_warmup_iterations=None,
        use_synthetic=False,
        display_every=100,
        mode='benchmark',
        target_duration=None,
        nb_items=26744,
        nb_users=138493,
        dup_mask=0.1,
        K=10):

    def model_fn(features, labels, mode):
        logits_out = tf.import_graph_def(frozen_graph,
                 input_map={'user_input:0': features["user_input"], 'item_input:0': features["item_input"]},
                 return_elements=['neumf/dense_3/BiasAdd:0'],
                 name='')
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                    predictions={'logits': logits_out[0]})
        if mode == tf.estimator.ModeKeys.EVAL:
            found_positive, dcg = compute_eval_metrics(logits_out[0], dup_mask, batch_size, K)
            hit_rate = tf.metrics.mean(found_positive, name='hit_rate')
            ndcg = tf.metrics.mean(dcg, name='ndcg')
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=dcg,
                eval_metric_ops={'hit_rate': hit_rate, 'ndcg': ndcg})

    def input_fn():
        if use_synthetic:
            items = [random.randint(1, nb_items) for _ in range(batch_size)]
            users = [random.randint(1, nb_users) for _ in range(batch_size)]
            with tf.device('/device:GPU:0'):
                items = tf.identity(items)
                users = tf.identity(users)
        else:
            data_path = os.path.join(data_dir, 'test_ratings.pickle')
            dataset = pd.read_pickle(data_path)
            users = dataset["user_id"]
            items = dataset["item_id"]
            print(type(users))
            users = users.astype('int32')
            items = items.astype('int32')
            user_dataset = tf.data.Dataset.from_tensor_slices(users)
            user_dataset = user_dataset.batch(batch_size)
            user_dataset = user_dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            user_dataset = user_dataset.repeat(count=1)
            user_iterator = user_dataset.make_one_shot_iterator()
            users = user_iterator.get_next()

            item_dataset = tf.data.Dataset.from_tensor_slices(items)
            item_dataset = item_dataset.batch(batch_size)
            item_dataset = item_dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            item_dataset = item_dataset.repeat(count=1)
            item_iterator = item_dataset.make_one_shot_iterator()
            items = item_iterator.get_next()
        return {"user_input": users, "item_input": items}, []
    
    if use_synthetic and num_iterations is None:
        num_iterations=1000
    
    if use_synthetic:
       num_records=num_iterations*batch_size
    else:
       data_path = os.path.join(data_dir, 'test_ratings.pickle')
       dataset = pd.read_pickle(data_path)
       users = dataset["user_id"]
       num_records = len(users)

    logger = LoggerHook(
        display_every=display_every,
        batch_size=batch_size,
        num_records=num_records)
    tf_config = tf.ConfigProto()
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(session_config=tf_config),
        model_dir='model_dir')
    results = {}    

    if mode == 'validation':
        results = estimator.evaluate(input_fn, steps=num_iterations, hooks=[logger])
    elif mode == 'benchmark':
        benchmark_hook = BenchmarkHook(target_duration=target_duration, iteration_limit=num_iterations)
        prediction_results = [p for p in estimator.predict(input_fn, predict_keys=["logits"], hooks=[logger, benchmark_hook])]
        print(prediction_results)
    else:
        raise ValueError("Mode must be either 'validation' or 'benchmark'")

    iter_times = np.array(logger.iter_times[num_warmup_iterations:])
    results['total_time'] = np.sum(iter_times)
    results['images_per_sec'] = np.mean(batch_size / iter_times)
    results['99th_percentile'] = np.percentile(iter_times, q=99, interpolation='lower') * 1000
    results['latency_mean'] = np.mean(iter_times) * 1000
    results['latency_median'] = np.median(iter_times) * 1000
    results['latency_min'] = np.min(iter_times) * 1000

    return results 


def input_data(use_synthetic=False,
               batch_size=1024,
               data_dir='/data/ml-20m/',
               num_iterations=None,
               nb_items=26744,
               nb_users=1388493):
    
    if use_synthetic and num_iterations is None:
        num_iterations = 10000

    if use_synthetic:
        items = [[random.randint(1, nb_items) for _ in range(batch_size)] for _ in range(num_iterations)]
        users = [[random.randint(1, nb_users) for _ in range(batch_size)] for _ in range(num_iterations)]
    else:
        if os.path.exists(data_dir):
             print("Using cached dataset: %s" % (data_dir))
        else:
             data_path = os.path.join(data_dir, 'test_ratings.pickle')
             dataset = pd.read_pickle(data_path)
             users = dataset["user_id"]
             items = dataset["item_id"]

    return items, users


def run_inference(frozen_graph,
                  use_synthetic=False,
                  mode='benchmark',
                  batch_size=1024,
                  data_dir='/data/ml-20m/',
                  num_iterations=10000,
                  num_warmup_iterations=2000,
                  nb_items=26744,
                  nb_users=1388493,
                  display_every=100):

    items, users = input_data(use_synthetic,
                              batch_size,
                              data_dir,
                              num_iterations,
                              nb_items,
                              nb_users)

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf.ConfigProto()) as tf_sess:
            tf.import_graph_def(frozen_graph, name='')
            for i in tf.get_default_graph().as_graph_def().node:
                print(i.name)
            output = tf_sess.graph.get_tensor_by_name('neumf/dense_3/BiasAdd:0')
            runtimes = []
            res = []

            for n in range(num_iterations):
                item = items[n]
                user = users[n]

                beg = time.time()
                r = tf_sess.run(output, feed_dict={'item_input:0': item, 'user_input:0': user})
                end = time.time()

                res.append(r)
                runtimes.append(end-beg)
                if n % display_every == 0:
                    print("    step %d/%d, iter_time(ms)=%.4f" % (
                        len(runtimes),
                        num_iterations,
                        np.mean(runtimes[(-1)*display_every:]) * 1000))
            print("throughput: %.1f" % 
                (batch_size * num_iterations/np.sum(runtimes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--use_synthetic', action='store_true',
            default=False,
            help='If set, one batch of random data is generated and used at every iteration.')
    parser.add_argument('--mode', choices=['validation', 'benchmark'],
            default='validation', help='Which mode to use (validation or benchmark)')
    parser.add_argument('--data_dir', type=str, default='/data/ml-20m/',
            help='Directory containing validation set csv files.')
    parser.add_argument('--calib_data_dir', type=str,
        help='Directory containing TFRecord files for calibrating int8.')
    parser.add_argument('--model_dir', type=str, default=None,
            help='Directory containing model checkpoint.')
    parser.add_argument('--use_trt', action='store_true',
            help='If set, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--use_dynamic_op', action='store_true',
        help='If set, TRT conversion will be done using dynamic op instead of statically.')
    parser.add_argument('--precision', type=str, 
            choices=['fp32', 'fp16', 'int8'], default='fp32',
            help='Precision mode to use. FP16 and INT8 only work in conjunction with --use_trt')
    parser.add_argument('--nb_items', type=int, default=26744,
            help='Number of items') 
    parser.add_argument('--nb_users', type=int, default=138493,
            help='Number of users')
    parser.add_argument('--batch_size', type=int, default=1024,
            help='Batch size')
    parser.add_argument('--minimum_segment_size', type=int, default=2,
        help='Minimum number of TF ops in a TRT engine')
    parser.add_argument('--num_iterations', type=int, default=None,
        help='How many iterations(batches) to evaluate. If not supplied, the whole set will be evaluated.')
    parser.add_argument('--num_warmup_iterations', type=int, default=50,
        help='Number of initial iterations skipped from timing')
    parser.add_argument('--num_calib_inputs', type=int, default=500,
        help='Number of inputs (e.g. images) used for calibration '
        '(last batch is skipped in case it is not full)')
    parser.add_argument('--max_workspace_size', type=int, default=(1<<32),
        help='workspace size in bytes')
    parser.add_argument('--display_every', type=int, default=100,
        help='Number of iterations executed between two consecutive display of metrics')
    parser.add_argument('--dup_mask', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--mf_dim', type=int, default=64)
    parser.add_argument('--mf_reg', type=int, default=64)
    parser.add_argument('--mlp_layer_sizes', default=[256, 256, 128, 64])
    parser.add_argument('--mlp_layer_regs', default=[.0, .0, .0, .0])

    args = parser.parse_args()
    if not args.use_synthetic and args.data_dir is None:
        raise ValueError("Data_dir is not provided")

    frozen_graph, num_nodes, times, graph_sizes = get_frozen_graph(
                       model_checkpoint=args.model_dir,
                       use_trt=args.use_trt,
                       use_dynamic_op=args.use_dynamic_op,
                       precision=args.precision,
                       batch_size=args.batch_size,
                       mf_dim=args.mf_dim,
                       mf_reg=args.mf_reg,
                       mlp_layer_sizes=args.mlp_layer_sizes,
                       mlp_layer_regs=args.mlp_layer_regs,
                       nb_items=args.nb_items,
                       nb_users=args.nb_users,
                       minimum_segment_size=args.minimum_segment_size,
                       calib_data_dir=args.calib_data_dir,
                       num_calib_inputs=args.num_calib_inputs,
                       use_synthetic=args.use_synthetic,
                       max_workspace_size=args.max_workspace_size
                       )                                  
                                  

    def print_dict(input_dict, str='', scale=None):
        for k, v in sorted(input_dict.items()):
            headline = '{}({}): '.format(str, k) if str else '{}: '.format(k)
            v = v * scale if scale else v
            print('{}{}'.format(headline, '%.1f'%v if type(v)==float else v))
    
    print_dict(num_nodes)
    print_dict(graph_sizes)
    print_dict(times)

    results = run(frozen_graph,
                  data_dir=args.data_dir,
                  batch_size=args.batch_size,
                  num_iterations=args.num_iterations,
                  num_warmup_iterations=args.num_warmup_iterations,
                  use_synthetic=args.use_synthetic,
                  display_every=args.display_every,
                  mode=args.mode,
                  target_duration=None,
                  nb_items=args.nb_items,
                  nb_users=args.nb_users,
                  dup_mask=args.dup_mask,
                  K=args.K)
    
    
    print_dict(results)

