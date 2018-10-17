import os

import rastervision as rv
from rastervision.backend import (Backend, BackendConfig, BackendConfigBuilder)
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg

NOOP_BACKEND = 'NOOP_BACKEND'
# Copied from tf_deeplab_config.py
# TODO: update
CHIP_OUTPUT_FILES = ['train-0.record', 'validation-0.record']
DEBUG_CHIP_OUTPUT_FILES = ['train.zip', 'validation.zip']


class NoopBackend(Backend):
    def process_scene_data(self, scene, data, tmp_dir):
        print(scene.id)
        return scene.id

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        print(training_results)

    def train(self, tmp_dir):
        pass

    def load_model(self, tmp_dir):
        pass

    def predict(self, chips, windows, tmp_dir):
        pass


class NoopBackendConfig(BackendConfig):
    def __init__(self):
        super().__init__(NOOP_BACKEND)

    def to_proto(self):
        msg = BackendConfigMsg(
            backend_type=self.backend_type, custom_config={})
        return msg

    def create_backend(self, task_config):
        return NoopBackend()

    def update_for_command(self, command_type, experiment_config, context=[]):
        # Copied from tf_deeplab_config.py
        # TODO: update
        conf, io_def = super().update_for_command(command_type,
                                                  experiment_config, context)
        if command_type == rv.CHIP:
            conf.training_data_uri = experiment_config.chip_uri
            outputs = list(
                map(lambda x: os.path.join(conf.training_data_uri, x),
                    CHIP_OUTPUT_FILES))
            io_def.add_outputs(outputs)
        if command_type == rv.TRAIN:
            conf.training_output_uri = experiment_config.train_uri
            inputs = list(
                map(lambda x: os.path.join(experiment_config.chip_uri, x),
                    CHIP_OUTPUT_FILES))
            io_def.add_inputs(inputs)

            conf.model_uri = os.path.join(conf.training_output_uri, 'model')
            io_def.add_output(conf.model_uri)

            '''
            # Set the fine tune checkpoint name to the experiment id
            if not conf.fine_tune_checkpoint_name:
                conf.fine_tune_checkpoint_name = experiment_config.id
            io_def.add_output(conf.fine_tune_checkpoint_name)
            '''
        if command_type in [rv.PREDICT, rv.BUNDLE]:
            if not conf.model_uri:
                io_def.add_missing('Missing model_uri.')
            else:
                io_def.add_input(conf.model_uri)

        return (conf, io_def)

    def save_bundle_files(self, bundle_dir):
        return (self, [])

    def load_bundle_files(self, bundle_dir):
        return self


class NoopBackendConfigBuilder(BackendConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(NOOP_BACKEND, NoopBackendConfig, {})

    @staticmethod
    def from_proto(msg):
        return NoopBackendConfigBuilder()

    def _applicable_tasks(self):
        return [rv.CHIP_CLASSIFICATION]

    def _process_task(self):
        return self


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.BACKEND, NOOP_BACKEND,
                                            NoopBackendConfigBuilder)
