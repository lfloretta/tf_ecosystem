
"""Compute stats for helping exploratory analysis"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import apache_beam as beam
import os
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.pyarrow_tf import pyarrow as pa

from tensorflow_metadata.proto.v0 import statistics_pb2

try:
    from utils import sql_queries
except ImportError as err:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils', 'src'))
    from utils import sql_queries


def compute_stats(bq_table,
                  stats_path,
                  max_row,
                  pipeline_args=None):
    """Computes statistics on the input data.

    Args:
        bq_table: BigQuery table in the format project.dataset.table
        step: (test, train, exploration, eval)
        stats_path: Directory in which stats are materialized.
        pipeline_args: additional DataflowRunner or DirectRunner args passed to the
          beam pipeline.
    """

    with beam.Pipeline(argv=pipeline_args) as pipeline:
        query = sql_queries.get_exploration_sql_query(bq_table, max_row=max_row)

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


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bq_table',
        help=('BigQuery Table with data to be analysed'))

    parser.add_argument(
        '--stats_path',
        help='Location for the computed stats to be materialized.')

    parser.add_argument(
        '--max_rows',
        default=None,
        help='Maximun number of record to export')

    known_args, pipeline_args = parser.parse_known_args()

    compute_stats(bq_table=known_args.bq_table,
                  stats_path=known_args.stats_path,
                  max_row=known_args.max_rows,
                  pipeline_args=pipeline_args)



if __name__ == '__main__':
    main()