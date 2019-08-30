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

echo Deploying a new version in CMLE...

REGION=us-central1
PROJECT=lf-ml-demo
TF_VERSION=1.14

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

if [ -z "${MODEL_NAME}" ]; then
  echo "You must specify a path to the saved model"
  exit 1
fi

if [ -z "${VERSION_NAME}" ]; then
  echo "You must specify a path to the saved model"
  exit 1
fi

if [ -z "${JOB_ID}" ]; then
  echo "You must specify a path to the saved model"
  exit 1
fi

HP_ID=`gcloud ai-platform jobs describe ${JOB_ID} --project $PROJECT --format 'value(trainingOutput.trials.trialId.slice(0))'`
JOB_DIR=`gcloud ai-platform jobs describe ${JOB_ID} --project $PROJECT --format 'value(trainingInput.jobDir)'`


MODEL_BINARIES=$(gsutil ls ${JOB_DIR}/${HP_ID}/serving-model-dir/export/model/ \
| sort | grep '\/[0-9]*\/$' | tail -n1)


gcloud ai-platform versions create ${VERSION_NAME} \
  --model ${MODEL_NAME} \
  --origin ${MODEL_BINARIES} \
  --runtime-version ${TF_VERSION} \
  --python-version 3.5 \
  --project ${PROJECT}