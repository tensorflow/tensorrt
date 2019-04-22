import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import time
import random
import numpy as np
from official.dataset import movielens

from neumf import ncf_model
from neumf import NeuMF
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
                     model_dtype=tf.float32,
                     mf_dim=64,
                     mf_reg=64,
                     mlp_layer_sizes=[256, 256, 128, 64],
                     mlp_layer_regs=[.0, .0, .0, .0],
                     nb_items=26744,
                     nb_users=138493):
    tf_config = tf.ConfigProto()
    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
            users = tf.placeholder(shape=(None,), dtype=tf.int32, name="user_input")
            items = tf.placeholder(shape=(None,), dtype=tf.int32, name="item_input")
            with tf.variable_scope("neumf"):
                logits = NeuMF(users, items, model_dtype, nb_users, nb_items, mf_dim, mf_reg, mlp_layer_sizes, mlp_layer_regs)
                if mode == "validation":
                    found_positive, dcg = compute_eval_metrics(logits, dup_mask, val_batch_size, K)
                    hit_rate = tf.metrics.mean(found_positive, name='hit_rate')
                    ndcg = tf.metrics.mean(dcg, name='ndcg')

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
    return frozen_graph


def optimize_model(frozen_graph,
                   use_trt=True,
                   precision_mode="FP16",
                   batch_size=128):
    if use_trt:
        trt_graph = trt.create_inference_graph(frozen_graph, ['neumf/dense_3/BiasAdd:0'], max_batch_size=batch_size, precision_mode=precision_mode)
    return trt_graph

def run(frozen_graph,
        data_dir='/data/cache/ml-20m',
        num_iterations=None,
        num_warmup_iterations=None,
        use_synthetic=False,
        display_every=100,
        mode='validation',
        target_duration=None):

    def model_fn(features, labels, mode):
        logits_out = tf.import_graph_def(frozen_graph,
                 input_map={'input': features},
                 return_elements=['logits:0'],
                 name='')
        found_possitive, dcg = compute_eval_metrics(logits, dup_mask, val_batch_size, K)
        hit_rate = tf.metrics.mean(found_positive, name='hit_rate')
        ndcg = tf.metrics.mean(found_positive, name='ndcg')
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                    predictions={'logits': logits_out})
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                eval_metrics_ops={'found_positive': found_possitive, 'ndcg': ndcg})

    def input_fn():
        if use_synthetic:
            items = [random.randint(1, nb_items) for _ in range(batch_size)]
            users = [random.randint(1, nb_users) for _ in range(batch_size)]
            with tf.devices('/device:GPU:0'):
                items = tf.identity(items)
                users = tf.identity(users)
        else:
            data_path = os.path.join(data_dir, 'test_ratings.pickle')
            dataset = pd.read_pickle(data_path)
            users = dataset["user_id"]
            items = dataset["item_id"]

            user_dataset = tf.data.Dataset_from_tensor_slices(users)
            user_dataset = user_dataset.batch(batch_size)
            user_dataset = user_dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            user_dataset = user_dataset.repeat(count=1)
            user_iterator = user_dataset.make_one_shot_iterator()
            users = user_iterator.get_next()

            item_dataset = tf.data.Dataset_from_tensor_slices(items)
            item_dataset = item_dataset.batch(batch_size)
            item_dataset = item_dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            item_dataset = item_dataset.repeat(count=1)
            item_iterator = item_dataset.make_one_shot_iterator()
            items = item_iterator.get_next()
        return items, users 


def input_data(use_synthetic=True,
               batch_size=128,
               data_dir="/data/cache/ml-20m",
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
                  use_synthetic=True,
                  mode='benchmark',
                  batch_size=128,
                  data_dir=None,
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
                        np.mean(runtimes[(-1)*display_every]) * 1000))
            print("throghput: %.1f" % 
                (batch_size * num_iterations/np.sum(runtimes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parse.add_argument('--use_synthetic', action='store_true',
            help='If set, one batch of random data is generated and used at every iteration.')
    parser.add_argument('--mode', choices=['validation', 'benchmark'],
            help='Which mode to use (validation or benchmark)')
    parser.add_argument('--data_dir', type=str, default=None,
            help='Directory containing validation set csv files.')
    parser.add_argument('--model_dir', type=str, default=None,
            help='Directory containing model checkpoint.')
    parser.add_argument('--use_trt', action='store_true',
            help='If set, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], default='fp32',
            help='Precision mode to use. FP16 and INT8 only work in conjunction with --use_trt')
    parser.add_argument('--nb_items', type=int, default=26744,
            help='Number of items') 
    parser.add_argument('--nb_users', type=int, default=1388493,
            help='Number of users')
    parser.add_argument('--batch_size', type=int, default=8,
            help='Batch size')
    parser.add_argument('--mf_dim', type=int, default=64)
    parser.add_argument('--mf_reg', type=int, default=64)
    parser.add_argument('--mlp_layer_sizes', default=[256, 256, 128, 64])
    parser.add_argument('--mlp_layer_regs', default=[.0, .0, .0, .0])

    args = parser.parse_args()
    if not args.use_synthetic and args.data_dir:
        raise ValueError("Data_dir is not provided")

    frozen_graph = get_frozen_graph()

    frozen_graph = optimize_model(frozen_graph)

    run_inference(frozen_graph)

