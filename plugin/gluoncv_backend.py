from os.path import join
import os
from copy import deepcopy
import shutil
import uuid
from google.protobuf import (json_format, struct_pb2)

import rastervision as rv
from rastervision.backend import (Backend, BackendConfig, BackendConfigBuilder)
from rastervision.utils.files import (make_dir, get_local_path, upload_or_copy,
                                      download_if_needed, start_sync,
                                      sync_to_dir, sync_from_dir)
from rastervision.utils.misc import save_img
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg

from plugin.train import gluoncv_train

GLUONCV_BACKEND = 'GLUONCV_BACKEND'
CHIP_OUTPUT_FILES = ['train-0.zip', 'validation-0.zip']

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

class ModelFiles(FileGroup):
    """Utilities for files produced when calling train."""

    def __init__(self, base_uri, tmp_dir, replace_model=False):
        """Create these model files.

        Args:
            base_uri: Base URI of the model files
            replace_model: If the model file exists, remove.
                           Used for the training step, to retrain
                           existing models.

        Returns:
            A new ModelFile instance.
        """
        FileGroup.__init__(self, base_uri, tmp_dir)
        self.model_uri = join(self.base_uri, 'model')
        self.log_uri = join(self.base_uri, 'log.csv')


        if replace_model:
            if os.path.exists(self.model_uri):
                os.remove(self.model_uri)
            if os.path.exists(self.log_uri):
                os.remove(self.log_uri)

    def download_backend_config(self, pretrained_model_uri, gcv_config,
                                dataset_files, class_map):
        from rastervision.protos.keras_classification.pipeline_pb2 \
            import PipelineConfig
        config = json_format.ParseDict(gcv_config, PipelineConfig())

        # Update config using local paths.
        config.trainer.options.output_dir = self.get_local_path(self.base_uri)
        config.model.model_path = self.get_local_path(self.model_uri)
        config.model.nb_classes = len(class_map)

        config.trainer.options.training_data_dir = \
            dataset_files.get_local_path(dataset_files.training_uri)
        config.trainer.options.validation_data_dir = \
            dataset_files.get_local_path(dataset_files.validation_uri)

        del config.trainer.options.class_names[:]
        config.trainer.options.class_names.extend(class_map.get_class_names())

        # Save the pretrained weights locally
        pretrained_model_path = None
        if pretrained_model_uri:
            pretrained_model_path = self.download_if_needed(
                pretrained_model_uri)

        # Save an updated copy of the config file.
        config_path = os.path.join(self.tmp_dir, 'gcv_config.json')
        config_str = json_format.MessageToJson(config)
        with open(config_path, 'w') as config_file:
            config_file.write(config_str)

        return (config_path, pretrained_model_path)



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
        dataset_files = DatasetFiles(self.config.training_data_uri, tmp_dir)
        dataset_files.download()

        model_files = ModelFiles(
            self.config.training_output_uri,
            tmp_dir)

        model_paths = model_files.download_backend_config(
            self.config.pretrained_model_uri, self.config.gcv_config,
            dataset_files, self.class_map)
        backend_config_path, pretrained_model_path = model_paths

        sync = start_sync(
            model_files.base_dir,
            self.config.training_output_uri,
            sync_interval=self.config.train_options.sync_interval)
        with sync:
            do_monitoring = self.config.train_options.do_monitoring
            gluoncv_train(backend_config_path, pretrained_model_path, do_monitoring)

    def load_model(self, tmp_dir):
        pass

    def predict(self, chips, windows, tmp_dir):
        pass


class GluonCVBackendConfig(BackendConfig):
    class TrainOptions:
        def __init__(self,
                     sync_interval=600,
                     do_monitoring=True,
                     replace_model=False):
            self.sync_interval = sync_interval
            self.do_monitoring = do_monitoring
            self.replace_model = replace_model

    def __init__(self,
                 gcv_config={},
                 train_options=None,
                 debug=False,
                 training_data_uri=None,
                 training_output_uri=None,
                 pretrained_model_uri=None,
                 model_uri=None):
        super().__init__(GLUONCV_BACKEND)

        self.gcv_config=gcv_config
        self.train_options = train_options
        self.debug = False

        self.training_data_uri = training_data_uri
        self.training_output_uri = training_output_uri
        self.pretrained_model_uri = pretrained_model_uri
        self.model_uri = model_uri

    def to_proto(self):
        struct = struct_pb2.Struct()
        struct['model_uri'] = self.model_uri
        struct['debug'] = self.debug
        struct['training_data_uri'] = self.training_data_uri
        struct['training_output_uri'] = self.training_output_uri

        msg = BackendConfigMsg(
            backend_type=self.backend_type, custom_config=struct)

        if self.pretrained_model_uri:
            msg.MergeFrom(
                BackendConfigMsg(
                    pretrained_model_uri=self.pretrained_model_uri))
        return msg

    def create_backend(self, task_config):
        return GluonCVBackend(self, task_config)

    def update_for_command(self, command_type, experiment_config, context=[]):
        conf, io_def = super().update_for_command(command_type,
                                                  experiment_config, context)
        if command_type == rv.ANALYZE:
            pass
        if command_type == rv.CHIP:
            if not conf.training_data_uri:
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
        if command_type in [rv.PREDICT, rv.BUNDLE]:
            if not conf.model_uri:
                io_def.add_missing('Missing model_uri.')
            else:
                io_def.add_input(conf.model_uri)

        return (conf, io_def)

    def save_bundle_files(self, bundle_dir):
        if not self.model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_path, base_name = self.bundle_file(self.model_uri, bundle_dir)
        new_config = self.to_builder() \
                         .with_model_uri(base_name) \
                         .build()
        return (new_config, [local_path])

    def load_bundle_files(self, bundle_dir):
        if not self.model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_model_uri = os.path.join(bundle_dir, self.model_uri)
        return self.to_builder() \
                   .with_model_uri(local_model_uri) \
                   .build()


class GluonCVBackendConfigBuilder(BackendConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'train_options': prev.train_options,
                'debug': prev.debug,
                'training_data_uri': prev.training_data_uri,
                'training_output_uri': prev.training_output_uri,
                'model_uri': prev.model_uri,
                'pretrained_model_uri': prev.pretrained_model_uri
            }
        self.require_task = prev is None
        super().__init__(GLUONCV_BACKEND, GluonCVBackendConfig, config)

    def from_proto(self, msg):
        b = super().from_proto(msg)
        conf = msg.custom_config
        b.require_task = None
        # b = b.with_debug(conf.debug)
        b = b.with_training_data_uri(conf['training_data_uri'])
        b = b.with_training_output_uri(conf['training_output_uri'])
        # TODO add other config options to b (model_uri, train_options, )


        return b

    def _applicable_tasks(self):
        return [rv.CHIP_CLASSIFICATION]

    def _process_task(self):
        return self

    def with_config(self,
                    config_mod,
                    ignore_missing_keys=False,
                    set_missing_keys=False):
        """Given a dict, modify the tensorflow pipeline configuration
           such that keys that are found recursively in the configuration
           are replaced with those values. TODO: better explination.
        """
        b = deepcopy(self)
        b.config_mods.append((config_mod, ignore_missing_keys,
                              set_missing_keys))
        return b

    def with_train_options(self,
                           sync_interval=600,
                           do_monitoring=True,
                           replace_model=False):
        """Sets the train options for this backend.

           Args:
              sync_interval: How often to sync output of training to
                             the cloud (in seconds).

              do_monitoring: Run process to monitor training (eg. Tensorboard)

              replace_model: Replace the model checkpoint if exists.
                             If false, this will continue training from
                             checkpoing if exists, if the backend allows for this.
        """
        b = deepcopy(self)
        b.config['train_options'] = GluonCVBackendConfig.TrainOptions(
            sync_interval, do_monitoring, replace_model)

        return b

    def with_training_data_uri(self, training_data_uri):
        """Whence comes the training data?

            Args:
                training_data_uri: The location of the training data.

        """
        b = deepcopy(self)
        b.config['training_data_uri'] = training_data_uri
        return b

    def with_training_output_uri(self, training_output_uri):
        """Whither goes the training output?

            Args:
                training_output_uri: The location where the training
                    output will be stored.

        """
        b = deepcopy(self)
        b.config['training_output_uri'] = training_output_uri
        return b

def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.BACKEND, GLUONCV_BACKEND,
                                            GluonCVBackendConfigBuilder)
