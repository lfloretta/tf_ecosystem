# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

"""
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
# import tensorflow_model_analysis as tfma
from tensorflow_transform import TFTransformOutput
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.saved import saved_transform_io


try:
    from utils import my_metadata
except ImportError as err:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'src'))
    from utils import my_metadata


# Functions for training
def make_training_input_fn(tft_output_dir,
                           filebase,
                           weight,
                           num_epochs=None,
                           shuffle=True,
                           batch_size=200,
                           buffer_size=None,
                           prefetch_buffer_size=1):
    """Creates an input function reading from transformed data.
    Args:
      tft_output_dir: Directory to read transformed data and metadata from and to
        write exported model to.
      filebase: Base filename (relative to `tft_output_dir`) of examples.
      num_epochs: int how many times through to read the data. If None will loop
        through data indefinitely
      shuffle: bool, whether or not to randomize the order of data. Controls
        randomization of both file order and line order within files.
      batch_size: Batch size
      buffer_size: Buffer size for the shuffle
      prefetch_buffer_size: Number of example to prefetch
    Returns:
      The input function for training or eval.
    """
    if buffer_size is None:
        buffer_size = 2 * batch_size + 1

    # Examples have already been transformed so we only need the feature_columns
    # to parse the single the tf.Record

    tft_output = TFTransformOutput(tft_output_dir)
    transformed_feature_spec = tft_output.transformed_feature_spec()

    def parser(record):
        """Help function to parse tf.Example."""
        parsed = tf.parse_single_example(record, transformed_feature_spec)
        label = parsed.pop(my_metadata.transformed_name(my_metadata.LABEL_KEY))

        if weight:
            #  create the weight
            ones = tf.ones_like(label, dtype=tf.float32)
            parsed['weight'] = tf.where(label == ones, tf.math.scalar_mul(weight, ones), ones)

        return parsed, label

    def input_fn():
        """Input function for training and eval."""
        files = tf.data.Dataset.list_files(os.path.join(tft_output_dir, filebase + '*'))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=16)
        dataset = dataset.cache()
        dataset = dataset.map(parser)

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset

    return input_fn


def make_serving_input_receiver_fn(tft_output_dir, schema_file):
    """Creates an input function from serving.
    Args:
      tft_output_dir: Directory to read transformed data and metadata from and to
        write exported model to.
    Returns:
      The input function for serving.
    """

    tft_output = TFTransformOutput(tft_output_dir)

    def input_fn():
        """Serving input function that reads raw data and applies transforms."""
        schema = my_metadata.read_schema(schema_file)
        raw_feature_spec = my_metadata.get_raw_feature_spec(schema)
        # Remove label since it is not available during serving.
        raw_feature_spec.pop(my_metadata.LABEL_KEY)

        raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            raw_feature_spec, default_batch_size=None)
        serving_input_receiver = raw_input_fn()

        transformed_features = tft_output.transform_raw_features(serving_input_receiver.features)

        return tf.estimator.export.ServingInputReceiver(
            transformed_features, serving_input_receiver.receiver_tensors)

    return input_fn


def make_serving_input_receiver_fn_json(tft_output_dir, schema_file):
    """Creates an input function from serving.
    Args:
      tft_output_dir: Directory to read transformed data and metadata from and to
        write exported model to.
      schema_file:
    Returns:
      The input function for serving.
    """

    tft_output = TFTransformOutput(tft_output_dir)

    def input_fn():
        """Serving input function that reads raw data and applies transforms."""
        schema = my_metadata.read_schema(schema_file)
        raw_feature_spec = my_metadata.get_raw_feature_spec(schema)
        # Remove label since it is not available during serving.
        raw_feature_spec.pop(my_metadata.LABEL_KEY)

        raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            raw_feature_spec, default_batch_size=None)

        raw_features, _, _ = raw_input_fn()

        receiver_tensors = {key: tf.placeholder(
            name=key, dtype=feature.dtype, shape=[None, 1]) for key, feature in raw_features.items()}
        # we are tranforming the raw_features with the graph written by
        # preprocess.py to transform_fn_io.TRANSFORM_FN_DIR and that was used to
        # write the tf records. This helps avoiding training/serving skew

        transformed_features = tft_output.transform_raw_features(raw_features)

        return tf.estimator.export.ServingInputReceiver(transformed_features,
                                                        receiver_tensors)

    return input_fn



# def make_eval_input_receiver_fn(tft_output_dir, schema_file, weight):
#     """Build everything needed for the tf-model-analysis to run the model.
#
#     Args:
#       tf_transform_dir: directory in which the tf-transform model was written
#         during the preprocessing step.
#       schema: the schema of the input data.
#
#     Returns:
#       EvalInputReceiver function, which contains:
#         - Tensorflow graph which parses raw untranformed features, applies the
#           tf-transform preprocessing operators.
#         - Set of raw, untransformed features.
#         - Label against which predictions will be compared.
#     """
#     tft_output = TFTransformOutput(tft_output_dir)
#
#     def eval_input_receiver_fn():
#         schema = my_metadata.read_schema(schema_file)
#         # Notice that the inputs are raw features, not transformed features here.
#         raw_feature_spec = my_metadata.get_raw_feature_spec(schema)
#
#         serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_example_tensor')
#
#         # Add a parse_example operator to the tensorflow graph, which will parse
#         # raw, untransformed, tf examples.
#         features = tf.parse_example(serialized_tf_example, raw_feature_spec)
#
#
#         transformed_features = tft_output.transform_raw_features(features)
#
#         # The key name MUST be 'examples'.
#         receiver_tensors = {'examples': serialized_tf_example}
#
#         # NOTE: Model is driven by transformed features (since training works on the
#         # materialized output of TFT, but slicing will happen on raw features.
#         features.update(transformed_features)
#
#         if weight:
#             ones = tf.ones_like(features[my_metadata.transformed_name(my_metadata.LABEL_KEY)], dtype=tf.float32)
#             features['weight'] = tf.where(
#                 features[my_metadata.transformed_name(my_metadata.LABEL_KEY)] == ones,
#                 tf.math.scalar_mul(weight, ones),
#                 ones)
#
#         return tfma.export.EvalInputReceiver(
#             features=features,
#             receiver_tensors=receiver_tensors,
#             labels=features[my_metadata.transformed_name(my_metadata.LABEL_KEY)])
#
#     return eval_input_receiver_fn
