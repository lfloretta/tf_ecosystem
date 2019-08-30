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
import base64
import json
import os
import sys

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

try:
    from utils import my_metadata
    from utils import sql_queries

except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils', 'src'))
    from utils import my_metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project',
        help=('GCP project'))

    parser.add_argument(
        '--model-name',
        help=('CMLE model name'))

    parser.add_argument(
        '--version-name',
        help=('CMLE version name'),
        default=None)

    parser.add_argument(
        '--schema-file', help='File holding the schema for the input data')

    parser.add_argument(
        '--examples-file', help='File holding the data for making the prediction')

    known_args, pipeline_args = parser.parse_known_args()

    schema = my_metadata.read_schema(known_args.schema_file)

    #  remove label
    filtered_features = [
        feature for feature in schema.feature if feature.name != my_metadata.LABEL_KEY]

    del schema.feature[:]
    schema.feature.extend(filtered_features)

    proto_coder = my_metadata.make_proto_coder(schema)
    raw_feature_spec = my_metadata.get_raw_feature_spec(schema)
    request_data = {}
    request_data['instances'] = []

    with open(known_args.examples_file, 'r') as f_in:
        for example in f_in.readlines():
            example_json = json.loads(example)
            example_clean_json = my_metadata.clean_raw_data_dict(example_json, raw_feature_spec=raw_feature_spec)
            example_serialized = proto_coder.encode(example_clean_json)
            request_data['instances'].append({'inputs': {'b64': base64.b64encode(example_serialized).decode()}})

    credentials = GoogleCredentials.get_application_default()
    api = discovery.build('ml', 'v1', credentials=credentials,
                          discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')

    if known_args.version_name:
        parent = 'projects/%s/models/%s/versions/%s' % (known_args.project,
                                                        known_args.model_name,
                                                        known_args.version_name)
    else:
        parent = 'projects/%s/models/%s/' % (known_args.project, known_args.model_name)

    response = api.projects().predict(body=request_data, name=parent).execute()
    print("response={0}".format(response))
