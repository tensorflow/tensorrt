# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import json
from .object_detection import build_model, download_dataset, optimize_model, benchmark_model


def test(test_config_path):
    """Runs an object detection test configuration
    
    This runs an object detection test configuration.  This involves
    
    1. Download and build a model architecture (or use cached).
    2. Optimize the model architecrue
    3. Benchmark the optimized model against a dataset
    4. (optional) Run assertions to check the benchmark output

    The input to this function is a JSON file which specifies the test
    configuration.

    example_test_config.json:

        {
            "model_config": { ... },
            "optimization_config": { ... },
            "benchmark_config": { ... },
            "assertions": [ ... ]
        }

    model_config: A dictionary of arguments passed to build_model, which
        specify the pre-optimized model architure.  The model will be passed
        to optimize_model.
    optimization_config: A dictionary of arguments passed to optimize_model.
        Please see help(optimize_model) for more details.
    benchmark_config: A dictionary of arguments passed to benchmark_model.
        Please see help(benchmark_model) for more details.
    assertions: A list of strings containing python code that will be 
        evaluated.  If the code returns false, an error will be thrown.  These
        assertions can reference any variables local to this 'test' function.
        Some useful values are

            statistics['map']
            statistics['avg_latency']
            statistics['avg_throughput']

    Args
    ----
        test_config_path: A string corresponding to the test configuration
            JSON file.
    """
    with open(args.test_config_path, 'r') as f:
        test_config = json.load(f)
        print(json.dumps(test_config, sort_keys=True, indent=4))

    frozen_graph = build_model(
        **test_config['model_config'])

    # optimize model using source model
    frozen_graph = optimize_model(
        frozen_graph,
        **test_config['optimization_config'])

    # benchmark optimized model
    statistics = benchmark_model(
        frozen_graph=frozen_graph,
        **test_config['benchmark_config'])

    # print some statistics to command line
    print_statistics = statistics
    if 'runtimes_ms' in print_statistics:
        print_statistics.pop('runtimes_ms')
    print(json.dumps(print_statistics, sort_keys=True, indent=4))

    # run assertions
    if 'assertions' in test_config:
        for a in test_config['assertions']:
            if not eval(a):
                raise AssertionError('ASSERTION FAILED: %s' % a)
            else:
                print('ASSERTION PASSED: %s' % a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_config_path',
        help='Path of JSON file containing test configuration.  Please'
             'see help(tftrt.examples.object_detection.test) for more information')
    args=parser.parse_args()
    test(args.test_config_path)
