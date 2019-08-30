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
"""

"""

import argparse
import json
import os

import tensorflow as tf
# import tensorflow_model_analysis as tfma
import trainer.input as input_module
import trainer.model as model

try:
    from utils import my_metadata
except ImportError as err:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'src'))
    from utils import my_metadata


def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters
    set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
            'index' in tf_config['task']):
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=[
                '/job:ps',
                '/job:worker/task:%d' % tf_config['task']['index']
            ])
    return None


def train_and_evaluate(hparams):
    """Run the training and evaluate using the high level API."""

    def train_input():
        """Input function returning batches from the training
        data set from training.
        """
        return input_module.make_training_input_fn(
            hparams.tft_output_dir,
            hparams.train_filebase,
            hparams.weight,
            num_epochs=hparams.num_epochs,
            batch_size=hparams.train_batch_size,
            buffer_size=hparams.buffer_size,
            prefetch_buffer_size=hparams.prefetch_buffer_size)

    def eval_input():
        """Input function returning the entire validation data
        set for evaluation. Shuffling is not required.
        """
        return input_module.make_training_input_fn(
            hparams.tft_output_dir,
            hparams.eval_filebase,
            hparams.weight,
            shuffle=False,
            batch_size=hparams.eval_batch_size,
            buffer_size=hparams.buffer_size,
            prefetch_buffer_size=hparams.prefetch_buffer_size)

    train_spec = tf.estimator.TrainSpec(
        train_input(), max_steps=hparams.train_steps)

    exporter = tf.estimator.FinalExporter(
        'model', input_module.make_serving_input_receiver_fn(
            hparams.tft_output_dir, hparams.schema_file))


    eval_spec = tf.estimator.EvalSpec(
        eval_input(),
        steps=hparams.eval_steps,
        exporters=[exporter],
        name='model-eval')

    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(hparams.job_dir, hparams.serving_model_dir),
        session_config=_get_session_config_from_env_var(),
        save_checkpoints_steps=999,
        keep_checkpoint_max=1)

    print('Model dir %s' % run_config.model_dir)

    estimator = model.build_estimator(
        tft_output_dir=hparams.tft_output_dir,
        embedding_size=hparams.embedding_size,
        # Construct layers sizes with exponential decay
        weight=hparams.weight,
        hidden_units=[
            max(2, int(hparams.first_layer_size * hparams.scale_factor**i))
            for i in range(hparams.num_layers)
        ],
        config=run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # tfma.export.export_eval_savedmodel(
    #     estimator=estimator,
    #     export_dir_base=os.path.join(hparams.job_dir, hparams.eval_model_dir),
    #     eval_input_receiver_fn=input_module.make_eval_input_receiver_fn(
    #         hparams.tft_output_dir, hparams.schema_file, hparams.weight)
    # )



if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    # Input Arguments
    PARSER.add_argument(
        '--tft-output-dir',
        help='GCS file or local paths to tf transform output',
        required=True
        )

    PARSER.add_argument(
        '--train-filebase',
        help='root of the files for training set during tft job',
        default='train-')

    PARSER.add_argument(
        '--serving-model-dir',
        help='Directory where to save trained model to be deployed',
        default='serving-model-dir')

    PARSER.add_argument(
        '--eval-model-dir',
        help='Directory where to save trained model for tfma',
        default='tfma-model-dir')

    PARSER.add_argument(
        '--eval-filebase',
        help='root of the files for training set during tft job',
        default='test-')

    PARSER.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models')

    PARSER.add_argument(
        '--schema-file',
        help='GCS location to tfdv schema',
        required=True)

    PARSER.add_argument(
        '--weight',
        help='Factor for weighting positive examples',
        type=float,
        default=None)

    PARSER.add_argument(
        '--buffer-size',
        help='Number of element for shuffling the batch',
        type=int
    )
    PARSER.add_argument(
        '--prefetch-buffer-size',
        help='Naximum number of input elements that will be buffered when prefetching',
        type=int,
        default=1)
    PARSER.add_argument(
        '--num-epochs',
        help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
        type=int)
    PARSER.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=500)
    PARSER.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=1000)
    PARSER.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for categorical columns',
        default=8,
        type=int)
    PARSER.add_argument(
        '--first-layer-size',
        help='Number of nodes in the first layer of the DNN',
        default=100,
        type=int)
    PARSER.add_argument(
        '--num-layers',
        help='Number of layers in the DNN',
        default=4,
        type=int)
    PARSER.add_argument(
        '--scale-factor',
        help='How quickly should the size of the layers in the DNN decay',
        default=0.7,
        type=float)
    PARSER.add_argument(
        '--train-steps',
        help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.""",
        default=100,
        type=int)
    PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=100,
        type=int)

    known_args, _ = PARSER.parse_known_args()

    hparams = tf.contrib.training.HParams(**known_args.__dict__)

    # Run the training job
    train_and_evaluate(hparams)
