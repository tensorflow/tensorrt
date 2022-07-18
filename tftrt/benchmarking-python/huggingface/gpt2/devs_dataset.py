#!# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import os
import tensorflow as tf
import dataloader

def get_dataset(data_dir, sequence_length=128, batch_size=32, vocab_size=512, use_random_data=False):
    if not use_random_data:
        dataset = dataloader.get_dataset_c4(
            data_dir,
            "",
            None,
            sequence_length,
            batch_size,
            vocab_size
        )

    else:
        tf.random.set_seed(12345)
        attention_mask = tf.random.uniform(
            shape=(1, sequence_length),
            maxval=vocab_size,
            dtype=tf.int32
        )
        ds_attention_mask = tf.data.Dataset.from_tensor_slices(attention_mask)

        decoder_attention_mask = tf.random.uniform(
            shape=(1, sequence_length),
            maxval=vocab_size,
            dtype=tf.int32
        )
        ds_decoder_attention_mask = tf.data.Dataset.from_tensor_slices(decoder_attention_mask)

        decoder_input_ids = tf.random.uniform(
            shape=(1, sequence_length),
            maxval=vocab_size,
            dtype=tf.int32
        )
        ds_decoder_input_ids = tf.data.Dataset.from_tensor_slices(decoder_input_ids)

        input_ids = tf.random.uniform(
            shape=(1, sequence_length),
            maxval=vocab_size,
            dtype=tf.int32
        )
        ds_input_ids = tf.data.Dataset.from_tensor_slices(input_ids)

        dataset = tf.data.Dataset.zip((
            ds_attention_mask,
            ds_decoder_attention_mask,
            ds_decoder_input_ids,
            ds_input_ids
        ))
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.take(count=1)  # loop over 1 batch
        dataset = dataset.cache()
        dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

dataset = get_dataset(data_dir="/data/c4/realnewslike/")

for idx, batch in enumerate(iter(dataset)):
    print(f"Step: {idx + 1}")
    if idx == 0:
        import pprint
        pprint.pprint(batch)