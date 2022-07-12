# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys

from functools import partial

import tensorflow as tf
import tensorflow_transform as tft

# Allow import of top level python files
import inspect

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)

benchmark_base_dir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, benchmark_base_dir)

from benchmark_args import BaseCommandLineAPI as CommandLineAPI
from benchmark_runner import BaseBenchmarkRunner


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%% IMPLEMENT MODEL-SPECIFIC FUNCTIONS HERE %%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


class BenchmarkRunner(BaseBenchmarkRunner):

    def get_dataset_batches(self):
        """Returns a list of batches of input samples.

        Each batch should be in the form [x, y], where
        x is a numpy array of the input samples for the batch, and
        y is a numpy array of the expected model outputs for the batch

        Returns:
        - dataset: a TF Dataset object
        - bypass_data_to_eval: any object type that will be passed unmodified to
                            `evaluate_result()`. If not necessary: `None`

        Note: script arguments can be accessed using `self._args.attr`
        """

        nvt_to_spark = {
            "ad_id": "ad_id",
            "ad_id_count": "ad_views_log_01scaled",
            "ad_id_ctr": "pop_ad_id",
            "advertiser_id": "ad_advertiser",
            "advertiser_id_ctr": "pop_advertiser_id",
            "campaign_id": "campaign_id",
            "campaign_id_ctr": "pop_campain_id",
            "clicked": "label",
            "display_id": "display_id",
            "document_id": "doc_event_id",
            "document_id_document_id_promo_sim_categories":
                "doc_event_doc_ad_sim_categories",
            "document_id_document_id_promo_sim_entities":
                "doc_event_doc_ad_sim_entities",
            "document_id_document_id_promo_sim_topics":
                "doc_event_doc_ad_sim_topics",
            "document_id_promo": "doc_id",
            "document_id_promo_count": "doc_views_log_01scaled",
            "document_id_promo_ctr": "pop_document_id",
            "geo_location": "event_geo_location",
            "geo_location_country": "event_country",
            "geo_location_state": "event_country_state",
            "platform": "event_platform",
            "publish_time_days_since_published":
                "doc_event_days_since_published_log_01scaled",
            "publish_time_promo_days_since_published":
                "doc_ad_days_since_published_log_01scaled",
            "publisher_id": "doc_event_publisher_id",
            "publisher_id_promo": "doc_ad_publisher_id",
            "publisher_id_promo_ctr": "pop_publisher_id",
            "source_id": "doc_event_source_id",
            "source_id_promo": "doc_ad_source_id",
            "source_id_promo_ctr": "pop_source_id",
        }

        spark_to_nvt = {val: key for key, val in nvt_to_spark.items()}

        def _consolidate_batch(elem):

            reshaped_and_renamed = {
                spark_to_nvt[key]: tf.reshape(value, [-1, value.shape[-1]])
                for key, value in elem.items()
                if key in spark_to_nvt
            }

            label = reshaped_and_renamed.pop("clicked")

            return reshaped_and_renamed, label

        if self._args.batch_size % 4096 != 0:
            raise ValueError(
                "Expected batch_size to be multiple of 4096, got "
                f"{self._args.batch_size}"
            )

        filepath_pattern = f"{self._args.data_dir}/eval/part*"

        feature_spec = tft.TFTransformOutput(self._args.data_dir)
        feature_spec = feature_spec.transformed_feature_spec()

        dataset = tf.data.Dataset.list_files(
            file_pattern=filepath_pattern, shuffle=False
        )

        dataset = tf.data.TFRecordDataset(
            filenames=dataset, num_parallel_reads=tf.data.AUTOTUNE
        )

        dataset = dataset.batch(
            batch_size=self._args.batch_size // 4096, drop_remainder=False
        )

        dataset = dataset.apply(
            transformation_func=tf.data.experimental.parse_example_dataset(
                features=feature_spec, num_parallel_calls=tf.data.AUTOTUNE
            )
        )

        dataset = dataset.map(
            map_func=partial(_consolidate_batch),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset, None

    def preprocess_model_inputs(self, data_batch):
        """This function prepare the `data_batch` generated from the dataset.
        Returns:
            x: input of the model
            y: data to be used for model evaluation

        Note: script arguments can be accessed using `self._args.attr`
        """

        x, y = data_batch

        y = {
            "y": y,
            "display_ids": x.pop('display_id')
        }

        return x, y

    def postprocess_model_outputs(self, predictions, expected):
        """Post process if needed the predictions and expected tensors. At the
        minimum, this function transforms all TF Tensors into a numpy arrays.
        Most models will not need to modify this function.

        Note: script arguments can be accessed using `self._args.attr`
        """
        expected["y"] = expected["y"].numpy()
        expected["display_ids"] = expected["display_ids"].numpy()

        return predictions.numpy(), expected


    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """

        display_ids = tf.reshape(expected["display_ids"], [-1])
        sorted_ids = tf.argsort(display_ids)

        display_ids = tf.gather(display_ids, indices=sorted_ids)

        predictions = tf.reshape(predictions["data"], [-1])
        predictions = tf.cast(predictions, tf.float64)
        predictions = tf.gather(predictions, indices=sorted_ids)

        labels = tf.reshape(expected["y"], [-1])
        labels = tf.gather(labels, indices=sorted_ids)

        _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(
            display_ids, out_idx=tf.int64
        )

        pad_length = 30 - tf.reduce_max(display_ids_ads_count)

        preds = tf.RaggedTensor.from_value_rowids(predictions,
                                                display_ids_idx).to_tensor()

        labels = tf.RaggedTensor.from_value_rowids(labels,
                                                display_ids_idx).to_tensor()

        labels_mask = tf.math.reduce_max(labels, 1)
        preds_masked = tf.boolean_mask(preds, labels_mask)
        labels_masked = tf.boolean_mask(labels, labels_mask)
        labels_masked = tf.argmax(labels_masked, axis=1, output_type=tf.int32)
        labels_masked = tf.reshape(labels_masked, [-1, 1])

        preds_masked = tf.pad(preds_masked, [(0, 0), (0, pad_length)])
        _, predictions_idx = tf.math.top_k(preds_masked, 12)
        indices = tf.math.equal(predictions_idx, labels_masked)
        indices_mask = tf.math.reduce_any(indices, 1)
        masked_indices = tf.boolean_mask(indices, indices_mask)

        res = tf.argmax(masked_indices, axis=1)
        ap_matrix = tf.divide(1, tf.add(res, 1))
        ap_sum = tf.reduce_sum(ap_matrix)
        shape = tf.cast(tf.shape(indices)[0], tf.float64)
        metric = (ap_sum / shape).numpy()

        return metric * 100, "Map@12"


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()
