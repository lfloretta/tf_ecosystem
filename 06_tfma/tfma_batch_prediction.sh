#!/bin/bash
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
set -u

echo Starting distributed TFMA  batch prediction...

REGION=us-central1
PROJECT=lf-ml-demo
BUCKET=lf-ml-demo-us-c1
BQ_TABLE=bigquery-public-data.chicago_taxi_trips.taxi_trips
TF_VERSION=1.14

while (($#)); do
   case $1 in
     "--job-id")
       shift
       JOB_ID="$1"
       shift
       ;;
     *)
       echo "Unknown argument: '$1'"
       exit 1
       ;;
   esac
done

if [ -z "${JOB_ID}" ]; then
  echo "You must specify a path to the saved model"
  exit 1
fi

HP_ID=`gcloud ai-platform jobs describe ${JOB_ID} --project $PROJECT --format 'value(trainingOutput.trials.trialId.slice(0))'`
JOB_DIR=`gcloud ai-platform jobs describe ${JOB_ID} --project $PROJECT --format 'value(trainingInput.jobDir)'`


EVAL_MODEL_DIR=$(gsutil ls ${JOB_DIR}/${HP_ID}/tfma-model-dir/ \
| sort | grep '\/[0-9]*\/$' | tail -n1)


# JOB_OUTPUT_PATH="gs://${BUCKET}/taxi-fare/$(date +%Y%m%d)"
JOB_OUTPUT_PATH="gs://${BUCKET}/taxi-fare/20190826"
TFDV_OUTPUT_PATH=$JOB_OUTPUT_PATH/tfdv_train_test_output

TFMA_OUTPUT_PATH=$JOB_OUTPUT_PATH/tfma_eval


JOB_ID="train-test-tfma-batch-$(date +%Y%m%d-%H%M%S)"
TEMP_PATH="gs://${BUCKET}/$JOB_ID/tmp"


python ./tfma_batch_prediction.py \
  --eval_model_dir $EVAL_MODEL_DIR \
  --eval_result_dir $TFMA_OUTPUT_PATH \
  --bq_table=${BQ_TABLE} \
  --max_rows=3000 \
  --schema_file=$TFDV_OUTPUT_PATH/schema.pbtxt \
  --project=${PROJECT} \
  --autoscaling_algorithm=THROUGHPUT_BASED \
  --region=${REGION} \
  --temp_location=$TEMP_PATH \
  --job_name=$JOB_ID \
  --setup_file=./setup.py \
  --save_main_session \
  --runner=DataflowRunner \
  --extra_package=../utils/dist/utils-0.1.tar.gz