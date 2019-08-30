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
"""Preprocessor applying tf.transform to the data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import apache_beam as beam
import tensorflow as tf

try:
    from utils import my_metadata
    from utils import sql_queries

except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils', 'src'))
    from utils import my_metadata
    from utils import sql_queries

import tensorflow_transform as transform
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.

    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse_to_dense(x.indices, [x.dense_shape[0], 1], x.values,
                           default_value),
        axis=1)


def transform_data(bq_table,
                   step,
                   schema_file,
                   working_dir,
                   outfile_prefix,
                   max_rows=None,
                   transform_dir=None,
                   pipeline_args=None):
    # todo : documentation
    """

    :param project:
    :param dataset:
    :param table:
    :param step:
    :param negative_sampling_ratio:
    :param train_cut:
    :param test_tenth:
    :param schema_file:
    :param working_dir:
    :param outfile_prefix:
    :param transform_dir:
    :param pipeline_args:
    :return:
    """

    def preprocessing_fn(inputs):
        """tf.transform's callback function for preprocessing inputs.

        Args:
          inputs: map from feature keys to raw not-yet-transformed features.

        Returns:
          Map from string feature key to transformed feature operations.
        """
        outputs = {}
        for key in my_metadata.NUMERIC_FEATURE_KEYS:
            # Preserve this feature as a dense float, setting nan's to the mean.
            outputs[my_metadata.transformed_name(key)] = transform.scale_to_z_score(_fill_in_missing(inputs[key]))

        for key in my_metadata.VOCAB_FEATURE_KEYS:
            # Build a vocabulary for this feature.
            outputs[my_metadata.transformed_name(key)] = transform.compute_and_apply_vocabulary(
                _fill_in_missing(inputs[key]),
                vocab_filename=my_metadata.transformed_name(key),
                num_oov_buckets=my_metadata.OOV_SIZE,
                top_k=my_metadata.VOCAB_SIZE
            )

        for key, hash_buckets in my_metadata.HASH_STRING_FEATURE_KEYS.items():
            outputs[my_metadata.transformed_name(key)] = transform.hash_strings(
                _fill_in_missing(inputs[key]),
                hash_buckets=hash_buckets
            )

        for key, nb_buckets in my_metadata.TO_BE_BUCKETIZED_FEATURE.items():
            outputs[my_metadata.transformed_name(key +'_bucketized')] = transform.bucketize(
                _fill_in_missing(inputs[key]), nb_buckets)


        # Was this passenger a big tipper?
        taxi_fare = _fill_in_missing(inputs[my_metadata.FARE_KEY])
        tips = _fill_in_missing(inputs[my_metadata.LABEL_KEY])
        outputs[my_metadata.transformed_name(my_metadata.LABEL_KEY)] = tf.where(
            tf.is_nan(taxi_fare),
            tf.cast(tf.zeros_like(taxi_fare), tf.int64),
            # Test if the tip was > 20% of the fare.
            tf.cast(
                tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))),
                tf.int64))

        return outputs

    schema = my_metadata.read_schema(schema_file)
    raw_feature_spec = my_metadata.get_raw_feature_spec(schema)
    raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
    raw_data_metadata = dataset_metadata.DatasetMetadata(raw_schema)

    with beam.Pipeline(argv=pipeline_args) as pipeline:
        with tft_beam.Context(temp_dir=working_dir):
            query = sql_queries.get_train_test_sql_query(bq_table, step, max_rows)
            raw_data = (
                    pipeline
                    | 'ReadBigQuery' >> beam.io.Read(
                beam.io.BigQuerySource(query=query, use_standard_sql=True))
                    | 'CleanData' >> beam.Map(
                my_metadata.clean_raw_data_dict, raw_feature_spec=raw_feature_spec))

            if transform_dir is None:
                transform_fn = (
                        (raw_data, raw_data_metadata)
                        | ('Analyze' >> tft_beam.AnalyzeDataset(preprocessing_fn)))

                _ = (
                        transform_fn
                        | ('WriteTransformFn' >>
                           tft_beam.WriteTransformFn(working_dir)))
            else:
                transform_fn = pipeline | tft_beam.ReadTransformFn(transform_dir)

            # Shuffling the data before materialization will improve Training
            # effectiveness downstream.
            shuffled_data = raw_data | 'RandomizeData' >> beam.transforms.Reshuffle()

            (transformed_data, transformed_metadata) = (
                    ((shuffled_data, raw_data_metadata), transform_fn)
                    | 'Transform' >> tft_beam.TransformDataset())

            coder = example_proto_coder.ExampleProtoCoder(transformed_metadata.schema)
            _ = (
                    transformed_data
                    | 'SerializeExamples' >> beam.Map(coder.encode)
                    | 'WriteExamples' >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, outfile_prefix))
            )


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bq_table',
        help=('BigQuery Table with data to be analysed'))

    parser.add_argument(
        '--step',
        choices=['test', 'train'],
        help='Step of the analyis, should be one of \'test\' or \'train\' ')

    parser.add_argument(
        '--max_rows',
        default=None,
        help='Maximun number of record to export')

    parser.add_argument(
        '--schema_file', help='File holding the schema for the input data')

    parser.add_argument(
        '--output_dir',
        help=('Directory in which transformed examples and function '
              'will be emitted.'))

    parser.add_argument(
        '--outfile_prefix',
        help='Filename prefix for emitted transformed examples')

    parser.add_argument(
        '--transform_dir',
        required=False,
        default=None,
        help='Directory in which the transform output is located')

    known_args, pipeline_args = parser.parse_known_args()

    transform_data(
        bq_table=known_args.bq_table,
        step=known_args.step,
        max_rows=known_args.max_rows,
        schema_file=known_args.schema_file,
        working_dir=known_args.output_dir,
        outfile_prefix=known_args.outfile_prefix,
        transform_dir=known_args.transform_dir,
        pipeline_args=pipeline_args)


if __name__ == '__main__':
    main()
