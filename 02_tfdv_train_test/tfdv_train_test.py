"""Prepare schema for train and test"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import apache_beam as beam
import os
import tensorflow as tf
import tensorflow_data_validation as tfdv

from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_data_validation.pyarrow_tf import pyarrow as pa
from tensorflow_metadata.proto.v0 import statistics_pb2

try:
    from utils import sql_queries
    from utils import my_metadata
except ImportError as err:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils', 'src'))
    from utils import sql_queries
    from utils import my_metadata


def compute_stats(bq_table,
                  step,
                  stats_path,
                  max_rows=None,
                  pipeline_args=None):
    # todo : update doc
    """Computes statistics on the input data.

    Args:
        table: BigQuery table
        step: (test, train)
        stats_path: Directory in which stats are materialized.
        pipeline_args: additional DataflowRunner or DirectRunner args passed to the
          beam pipeline.
    """

    with beam.Pipeline(argv=pipeline_args) as pipeline:
        query = sql_queries.get_train_test_sql_query(bq_table, step, max_rows)

        raw_data = (
                pipeline
                | 'ReadBigQuery' >> beam.io.Read(
            beam.io.BigQuerySource(query=query, use_standard_sql=True))
                | 'ConvertToTFDVInput' >> beam.Map(
            lambda x: pa.Table.from_pydict(
                {key: [[x[key]]] for key in x if x[key] is not None})))

        _ = (
                raw_data
                | 'GenerateStatistics' >> tfdv.GenerateStatistics()
                | 'WriteStatsOutput' >> beam.io.WriteToTFRecord(
            stats_path,
            shard_name_template='',
            coder=beam.coders.ProtoCoder(
                statistics_pb2.DatasetFeatureStatisticsList)))


def infer_schema(stats_path, schema_path):
    """Infers a schema from stats in stats_path.

    Args:
      stats_path: Location of the stats used to infer the schema.
      schema_path: Location where the inferred schema is materialized.
    """
    # Infering schema from statistics
    schema = tfdv.infer_schema(tfdv.load_statistics(stats_path), infer_feature_shape=False)

    # Writing schema to output path
    file_io.write_string_to_file(schema_path, text_format.MessageToString(schema))

def validate_stats(stats_path, schema_path, anomalies_path):
    """Validates the statistics against the schema and materializes anomalies.

    Args:
      stats_path: Location of the stats used to infer the schema.
      schema_path: Location of the schema to be used for validation.
      anomalies_path: Location where the detected anomalies are materialized.
    """
    # Validating schema against the computed statistics
    schema = my_metadata.read_schema(schema_path)

    stats = tfdv.load_statistics(stats_path)
    anomalies = tfdv.validate_statistics(stats, schema)

    # Writing anomalies to anomalies path to
    file_io.write_string_to_file(anomalies_path,
                                 text_format.MessageToString(anomalies))


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
        '--stats_path',
        help='Location for the computed stats to be materialized.')

    parser.add_argument(
        '--schema_path',
        help='Location for the computed schema to be materialized.')

    parser.add_argument(
        '--anomalies_path',
        help='Location for the anomalies to be materialized.',
    )

    parser.add_argument(
        '--composer_dataflow_save_main_session',
        type=bool,
        default=False,
        help='Help flag to set --save-main-session for dataflow running on composer'
    )

    known_args, pipeline_args = parser.parse_known_args()

    if known_args.composer_dataflow_save_main_session:
        pipeline_args.append('--save_main_session')

    compute_stats(bq_table=known_args.bq_table,
                  step=known_args.step,
                  stats_path=known_args.stats_path,
                  max_rows=known_args.max_rows,
                  pipeline_args=pipeline_args)

    if known_args.schema_path and known_args.step == 'train':
        infer_schema(stats_path=known_args.stats_path, schema_path=known_args.schema_path)


    if known_args.schema_path and known_args.anomalies_path:
        validate_stats(stats_path=known_args.stats_path,
                       schema_path=known_args.schema_path,
                       anomalies_path=known_args.anomalies_path)



if __name__ == '__main__':
    main()