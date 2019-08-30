#!/bin/bash
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
set -u

echo Starting distributed TFT preprocessing...

PROJECT=lf-ml-demo
BUCKET=lf-ml-demo-us-c1
BQ_TABLE=bigquery-public-data.chicago_taxi_trips.taxi_trips
REGION=us-central1

# JOB_OUTPUT_PATH="gs://${BUCKET}/taxi-fare/$(date +%Y%m%d)"
JOB_OUTPUT_PATH="gs://${BUCKET}/taxi-fare/20190826"
TFDV_OUTPUT_PATH=$JOB_OUTPUT_PATH/tfdv_train_test_output

TFT_OUTPUT_PATH="$JOB_OUTPUT_PATH/tft_train_test"

JOB_ID="taxi-fare-tft-train-$(date +%Y%m%d-%H%M%S)"
TEMP_PATH="gs://${BUCKET}/$JOB_ID/tmp"


echo ...for training...

python3 ./preprocess.py \
  --bq_table=${BQ_TABLE} \
  --step=train \
  --max_rows=10000 \
  --schema_file=$TFDV_OUTPUT_PATH/schema.pbtxt \
  --output_dir=$TFT_OUTPUT_PATH \
  --outfile_prefix=train- \
  --project=${PROJECT} \
  --autoscaling_algorithm=THROUGHPUT_BASED \
  --region=${REGION} \
  --temp_location=$TEMP_PATH \
  --job_name=$JOB_ID \
  --setup_file=./setup.py \
  --runner=DataflowRunner \
  --extra_package=../utils/dist/utils-0.1.tar.gz

echo ...for evaluating...


JOB_ID="taxi-fare-tft-test-$(date +%Y%m%d-%H%M%S)"

python ./preprocess.py \
  --bq_table=${BQ_TABLE} \
  --step=test \
  --max_rows=3000 \
  --schema_file=$TFDV_OUTPUT_PATH/schema.pbtxt \
  --output_dir=$TFT_OUTPUT_PATH \
  --outfile_prefix=test- \
  --transform_dir=$TFT_OUTPUT_PATH \
  --project=${PROJECT} \
  --autoscaling_algorithm=THROUGHPUT_BASED \
  --region=${REGION} \
  --temp_location=$TEMP_PATH \
  --job_name=$JOB_ID \
  --setup_file=./setup.py \
  --runner=DataflowRunner \
  --extra_package=../utils/dist/utils-0.1.tar.gz

