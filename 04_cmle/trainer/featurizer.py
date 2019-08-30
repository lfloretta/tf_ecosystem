# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Defines a Wide + Deep model for classification on structured data.
"""
from __future__ import division
from __future__ import print_function


import os
import tensorflow as tf

from tensorflow_transform import TFTransformOutput

try:
    from utils import my_metadata
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'src'))
    from utils import my_metadata



def get_deep_and_wide_columns(tft_transform_dir, embedding_size=8):
    """Creates deep and wide feature_column lists.
    Args:
            tf_transform_dir: (str), directory in which the tf-transform model was written
                                     during the preprocessing step.
            embedding_size: (int), the number of dimensions used to represent categorical
                                   features when providing them as inputs to the DNN.
    Returns:
            [tf.feature_column],[tf.feature_column]: deep and wide feature_column lists.
    """

    tft_output = TFTransformOutput(tft_transform_dir)
    transformed_feature_spec = tft_output.transformed_feature_spec()

    transformed_feature_spec.pop(my_metadata.transformed_name(my_metadata.LABEL_KEY))

    deep_columns = {}
    wide_columns = {}

    for transformed_key, tensor in transformed_feature_spec.items():
        #  Separate features by deep and wide
        if transformed_key in my_metadata.transformed_names(my_metadata.VOCAB_FEATURE_KEYS):
            if transformed_key not in my_metadata.transformed_names(my_metadata.CATEGORICAL_FEATURE_KEYS_TO_BE_REMOVED):
                wide_columns[transformed_key] = tf.feature_column.categorical_column_with_identity(
                    key=transformed_key,
                    num_buckets=tft_output.vocabulary_size_by_name(transformed_key) + my_metadata.OOV_SIZE
                )

        elif transformed_key in my_metadata.transformed_names(my_metadata.HASH_STRING_FEATURE_KEYS):
            if transformed_key not in my_metadata.transformed_names(my_metadata.CATEGORICAL_FEATURE_KEYS_TO_BE_REMOVED):
                wide_columns[transformed_key] = tf.feature_column.categorical_column_with_identity(
                    key=transformed_key,
                    num_buckets=my_metadata.HASH_STRING_FEATURE_KEYS[my_metadata.original_name(transformed_key)]
                )

        elif transformed_key in my_metadata.transformed_names(my_metadata.NUMERIC_FEATURE_KEYS):
            if transformed_key not in my_metadata.transformed_names(my_metadata.NUMERIC_FEATURE_KEYS_TO_BE_REMOVED):
                deep_columns[transformed_key] = tf.feature_column.numeric_column(transformed_key)

        elif (
                (transformed_key.endswith(my_metadata.transformed_name('_bucketized'))
                    and transformed_key.replace(
                            my_metadata.transformed_name('_bucketized'), '') in my_metadata.TO_BE_BUCKETIZED_FEATURE)):
            wide_columns[transformed_key] = tf.feature_column.categorical_column_with_identity(
                key=transformed_key,
                num_buckets=tft_output.num_buckets_for_transformed_feature(transformed_key)
            )

        else:
            raise LookupError('The couple (%s, %s) is not consistent with utils.my_metadata' % (key, tensor))

    # # #  creating new categorical features
    wide_columns.update(
        {
        'pickup_latitude_bucketized_xf_x_pickup_longitude_bucketized_xf' : tf.feature_column.crossed_column(
            ['pickup_latitude_bucketized_xf', 'pickup_longitude_bucketized_xf'],
            hash_bucket_size=int(1e3)),
        }
    )
    #
    # # creating new numeric features from categorical features
    deep_columns.update(
        {
            # Use indicator columns for low dimensional vocabularies
            'trip_start_day_xf_indicator': tf.feature_column.indicator_column(wide_columns['trip_start_day_xf']),

            # Use embedding columns for high dimensional vocabularies
            'company_xf_embedding':  tf.feature_column.embedding_column(
                wide_columns['company_xf'], dimension=embedding_size)
        }
    )

    return deep_columns.values(), wide_columns.values()

