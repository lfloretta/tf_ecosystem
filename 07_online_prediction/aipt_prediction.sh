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

echo Prediction from in AI Platform Prediction...

PROJECT=lf-ml-demo
BUCKET=lf-ml-demo-us-c1

# JOB_OUTPUT_PATH="gs://${BUCKET}/taxi-fare/$(date +%Y%m%d)"
JOB_OUTPUT_PATH="gs://${BUCKET}/taxi-fare/20190402"

TFDV_OUTPUT_PATH=$JOB_OUTPUT_PATH/tfdv_train_test_output


while (($#)); do
   case $1 in
     "--model-name")
       shift
       MODEL_NAME="$1"
       shift
       ;;
     "--version-name")
       shift
       VERSION_NAME="$1"
       shift
       ;;
     "--examples-file")
       shift
       EXAMPLES_FILE="$1"
       shift
       ;;
     *)
       echo "Unknown argument: '$1'"
       exit 1
       ;;
   esac
done

python3 ./aipt_prediction.py --project $PROJECT \
    --model-name $MODEL_NAME \
    --version-name $VERSION_NAME \
    --schema-file $TFDV_OUTPUT_PATH/schema.pbtxt \
    --examples-file $EXAMPLES_FILE
