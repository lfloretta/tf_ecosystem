# Copyright 2018 Google LLC
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
"""Runs a batch job for performing Tensorflow Model Analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import apache_beam as beam
import tensorflow as tf

import tensorflow_model_analysis as tfma

try:
    from utils import my_metadata
except ImportError as err:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils', 'src'))
    from utils import my_metadata
    from utils import sql_queries


def process_tfma(eval_model_dir=None,
                 eval_result_dir=None,
                 bq_table=None,
                 max_rows=None,
                 schema_file=None,
                 pipeline_args=None):
    """Runs a batch job to evaluate the eval_model against the given input.
    :param eval_model_dir:
    :param eval_result_dir:
    :param bq_table:
    :param max_rows:
    :param max_rows:
    :param pipeline_args:
    :return:
    """

    slice_spec = [
        tfma.slicer.SingleSliceSpec()
    ]

    for slice in my_metadata.TFMA_SLICERS:
        slice_spec.append(tfma.slicer.SingleSliceSpec(columns=slice[0], features=slice[1]))

    schema = my_metadata.read_schema(schema_file)

    eval_shared_model = tfma.default_eval_shared_model(
        eval_saved_model_path=eval_model_dir,
        add_metrics_callbacks=[
            tfma.post_export_metrics.calibration_plot_and_prediction_histogram(),
            tfma.post_export_metrics.auc_plots(),
            tfma.post_export_metrics.auc()
        ])

    with beam.Pipeline(argv=pipeline_args) as pipeline:

        query = sql_queries.get_tfma_sql_query(bq_table, max_rows)
        raw_feature_spec = my_metadata.get_raw_feature_spec(schema)

        raw_data = (
                pipeline
                | 'ReadBigQuery' >> beam.io.Read(
            beam.io.BigQuerySource(query=query, use_standard_sql=True))
                | 'CleanData' >>
                beam.Map(lambda x: (my_metadata.clean_raw_data_dict(x, raw_feature_spec))))

        # Examples must be in clean tf-example format.
        coder = my_metadata.make_proto_coder(schema)

        _ = (
                raw_data
                | 'ToSerializedTFExample' >> beam.Map(coder.encode)
                |
                'ExtractEvaluateAndWriteResults' >> tfma.ExtractEvaluateAndWriteResults(
            eval_shared_model=eval_shared_model,
            slice_spec=slice_spec,
            output_path=eval_result_dir))


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--eval_model_dir',
        help='Input path to the model which will be evaluated.',
        required=True
    )
    parser.add_argument(
        '--eval_result_dir',
        help='Output directory in which the model analysis result is written.',
        required=True
    )
    parser.add_argument(
        '--bq_table',
        help=('BigQuery Table with data to be analysed'))
    parser.add_argument(
        '--max_rows',
        default=None,
        type=int,
        help='Limit number of rows')
    parser.add_argument(
        '--schema_file',
        help='File holding the schema for the input data')

    parser.add_argument(
        '--composer_dataflow_save_main_session',
        type=bool,
        default=False,
        help='Help flag to set --save-main-session for dataflow running on composer'
    )

    known_args, pipeline_args = parser.parse_known_args()

    if known_args.composer_dataflow_save_main_session:
        pipeline_args.append('--save_main_session')

    process_tfma(
        eval_model_dir=known_args.eval_model_dir,
        eval_result_dir=known_args.eval_result_dir,
        bq_table=known_args.bq_table,
        max_rows=known_args.max_rows,
        schema_file=known_args.schema_file,
        pipeline_args=pipeline_args)


if __name__ == '__main__':
    main()
