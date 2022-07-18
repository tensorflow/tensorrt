#!# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf
import tensorflow_text as text  # Registers the ops.


def get_dataset_cola(batch_size, filename, tokenizer_dir, sequence_length):
    # This function is used to load data from the file in filename
    preprocessor = tf.saved_model.load(tokenizer_dir)

    def extract_actual_text_fn(line):
        line = tf.strings.regex_replace(line, "[0-9]\t", "")
        line = tf.strings.regex_replace(line, "[0-9]", "")
        encoder_ips = preprocessor([line])
        encoder_ips = {
            key: tf.squeeze(value) for key, value in encoder_ips.items()
        }
        return encoder_ips

    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(
        extract_actual_text_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset
