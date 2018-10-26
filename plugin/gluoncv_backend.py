from os.path import join
import os
import shutil
import uuid

import rastervision as rv
from rastervision.backend import (Backend, BackendConfig, BackendConfigBuilder)
from rastervision.utils.files import (make_dir, get_local_path, upload_or_copy,
                                      download_if_needed, start_sync,
                                      sync_to_dir, sync_from_dir)
from rastervision.utils.misc import save_img
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg

GLUONCV_BACKEND = 'GLUONCV_BACKEND'
CHIP_OUTPUT_FILES = ['train-0.record', 'validation-0.record']

class FileGroup(object):
    def __init__(self, base_uri, tmp_dir):
        self.tmp_dir = tmp_dir
        self.base_uri = base_uri
        self.base_dir = self.get_local_path(base_uri)

        make_dir(self.base_dir)


    def get_local_path(self, uri):
        return get_local_path(uri, self.tmp_dir)

    def upload_or_copy(self, uri):
        upload_or_copy(self.get_local_path(uri), uri)

    def download_if_needed(self, uri):
        return download_if_needed(uri, self.tmp_dir)


class DatasetFiles(FileGroup):
    """Utilities for files produced when calling convert_training_data."""

    def __init__(self, base_uri, tmp_dir):
        FileGroup.__init__(self, base_uri, tmp_dir)

        self.training_uri = join(base_uri, 'training')
        make_dir(self.get_local_path(self.training_uri))
        self.training_zip_uri = join(base_uri, 'training.zip')

        self.validation_uri = join(base_uri, 'validation')
        make_dir(self.get_local_path(self.validation_uri))
        self.validation_zip_uri = join(base_uri, 'validation.zip')

        self.scratch_uri = join(base_uri, 'scratch')
        make_dir(self.get_local_path(self.scratch_uri))

    def download(self):
        def _download(data_zip_uri):
            data_zip_path = self.download_if_needed(data_zip_uri)
            data_dir = os.path.splitext(data_zip_path)[0]
            shutil.unpack_archive(data_zip_path, data_dir)

        _download(self.training_zip_uri)
        _download(self.validation_zip_uri)

    def upload(self):
        def _upload(data_uri):
            data_dir = self.get_local_path(data_uri)
            shutil.make_archive(data_dir, 'zip', data_dir)
            self.upload_or_copy(data_uri + '.zip')

        _upload(self.training_uri)
        _upload(self.validation_uri)


class GluonCVBackend(Backend):
    def __init__(self, backend_config, task_config):
        self.model = None
        self.config = backend_config
        self.class_map = task_config.class_map

    def process_scene_data(self, scene, data, tmp_dir):
        """Process each scene's training data

        Args:
            scene: Scene
            data: TrainingData

        Returns:
            dictionary of Scene's classes and corresponding local directory
                path
        """
        dataset_files = DatasetFiles(self.config.training_data_uri, tmp_dir)

        scratch_dir = dataset_files.get_local_path(dataset_files.scratch_uri)

        scene_dir = join(scratch_dir, '{}-{}'.format(scene.id, uuid.uuid4()))
        class_dirs = {}

        for chip_idx, (chip, window, labels) in enumerate(data):
            class_id = labels.get_cell_class_id(window)

            if class_id == None:
                continue
            class_name = self.class_map.get_by_id(class_id).name
            class_dir = join(scene_dir, class_name)
            make_dir(class_dir)
            class_dirs[class_name] = class_dir
            chip_name = '{}.png'.format(chip_idx)
            chip_path = join(class_dir, chip_name)
            save_img(chip, chip_path)

        return class_dirs

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """After all scenes have been processed, collect all the images of
        each class across all scenes

        Args:
            training_results: list of dictionaries of training scenes'
                classes and corresponding local directory path
            validation_results: list of dictionaries of validation scenes'
                classes and corresponding local directory path
        """
        dataset_files = DatasetFiles(self.config.training_data_uri, tmp_dir)
        training_dir = dataset_files.get_local_path(dataset_files.training_uri)
        validation_dir = dataset_files.get_local_path(
            dataset_files.validation_uri)

        def _merge_training_results(scene_class_dirs, output_dir):
            for class_name in self.class_map.get_class_names():
                class_dir = join(output_dir, class_name)
                make_dir(class_dir)

            chip_ind = 0
            for scene_class_dir in scene_class_dirs:
                for class_name, src_class_dir in scene_class_dir.items():
                    dst_class_dir = join(output_dir, class_name)
                    src_class_files = [
                        join(src_class_dir, class_file)
                        for class_file in os.listdir(src_class_dir)
                    ]
                    for src_class_file in src_class_files:
                        dst_class_file = join(dst_class_dir,
                                              '{}.png'.format(chip_ind))
                        shutil.move(src_class_file, dst_class_file)
                        chip_ind += 1
        _merge_training_results(training_results, training_dir)
        _merge_training_results(validation_results, validation_dir)
        dataset_files.upload()

    def train(self, tmp_dir):
        pass

    def load_model(self, tmp_dir):
        pass

    def predict(self, chips, windows, tmp_dir):
        pass


class GluonCVBackendConfig(BackendConfig):
    def __init__(self, debug=False, training_data_uri=None):
        super().__init__(GLUONCV_BACKEND)

        self.debug = False

    def to_proto(self):
        msg = BackendConfigMsg(
            backend_type=self.backend_type, custom_config={})
        return msg

    def create_backend(self, task_config):
        return GluonCVBackend(self, task_config)

    def update_for_command(self, command_type, experiment_config, context=[]):
        conf, io_def = super().update_for_command(command_type,
                                                  experiment_config, context)
        if command_type == rv.ANALYZE:
            pass
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


class GluonCVBackendConfigBuilder(BackendConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'training_data_uri': prev.training_data_uri
            }
        super().__init__(GLUONCV_BACKEND, GluonCVBackendConfig, {})


    @staticmethod
    def from_proto(msg):
        return GluonCVBackendConfigBuilder()

    def _applicable_tasks(self):
        return [rv.CHIP_CLASSIFICATION]

    def _process_task(self):
        return self


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.BACKEND, GLUONCV_BACKEND,
                                            GluonCVBackendConfigBuilder)
