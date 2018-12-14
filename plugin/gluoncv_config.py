from os.path import join
from copy import deepcopy
from google.protobuf import struct_pb2

import rastervision as rv
from rastervision.backend import (BackendConfig, BackendConfigBuilder)
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg

from plugin.gluoncv_backend import GluonCVBackend

GLUONCV_BACKEND = 'GLUONCV_BACKEND'
CHIP_OUTPUT_FILES = ['training.zip', 'validation.zip']


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

        if train_options is None:
            train_options = GluonCVBackendConfig.TrainOptions()

        super().__init__(GLUONCV_BACKEND)

        self.gcv_config = gcv_config
        self.train_options = train_options
        self.debug = False
        self.training_data_uri = training_data_uri
        self.training_output_uri = training_output_uri
        self.pretrained_model_uri = pretrained_model_uri
        self.model_uri = model_uri

    def to_proto(self):
        struct = struct_pb2.Struct()
        struct['sync_interval'] = self.train_options.sync_interval
        struct['do_monitoring'] = self.train_options.do_monitoring
        struct['replace_model'] = self.train_options.replace_model
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

    def update_for_command(self, command_type, experiment_config,
                           context=None, io_def=None):
        io_def = super().update_for_command(command_type, experiment_config,
                                            context, io_def)
        if command_type == rv.ANALYZE:
            pass
        if command_type == rv.CHIP:
            if not self.training_data_uri:
                self.training_data_uri = experiment_config.chip_uri
            outputs = list(
                map(lambda x: join(self.training_data_uri, x),
                    CHIP_OUTPUT_FILES))
            io_def.add_outputs(outputs)

        if command_type == rv.TRAIN:
            if not self.training_data_uri:
                io_def.add_missing('Missing training_data_uri.')
            else:
                inputs = list(map(
                            lambda x: join(self.training_data_uri, x),
                            CHIP_OUTPUT_FILES))
                io_def.add_inputs(inputs)
            if not self.training_output_uri:
                self.training_output_uri = experiment_config.train_uri
            if not self.model_uri:
                self.model_uri = join(self.training_output_uri,
                                      'model')
            io_def.add_output(self.model_uri)

        if command_type in [rv.PREDICT, rv.BUNDLE]:
            if not self.model_uri:
                io_def.add_missing('Missing model_uri.')
            else:
                io_def.add_input(self.model_uri)

        return io_def

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
        local_model_uri = join(bundle_dir, self.model_uri)
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

        b = b.with_training_data_uri(conf['training_data_uri'])
        b = b.with_training_output_uri(conf['training_output_uri'])
        b = b.with_model_uri(conf['model_uri'])

        return b

    def _applicable_tasks(self):
        return [rv.CHIP_CLASSIFICATION]

    def _process_task(self):
        return self

    def with_config(self,
                    config_mod,
                    ignore_missing_keys=False,
                    set_missing_keys=False):
        """Given a dict, modify the pipeline configuration
           such that keys that are found recursively in the configuration
           are replaced with those values.
        """
        # Note: this method is not fully implemented. It is being left here
        # because it may be useful for anyone picking up this project. For
        # now, it will just raise this error.
        raise NotImplementedError('.with_config() option not fully '
                                  'implemented.')
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

              replace_model: Replace the model checkpoint if exists.
                             If false, this will continue training from
                             checkpoing if exists, if the backend allows
                             for this.
        """
        # Note: this method is not fully implemented. It is being left here
        # because it may be useful for anyone picking up this project. For
        # now, it will just raise this error.
        raise NotImplementedError('.with_train_options() not fully '
                                  'implemented.')
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

    def with_model_uri(self, model_uri):
        """Defines the name of the model file that will be created for
        this model after training.

        """
        b = deepcopy(self)
        b.config['model_uri'] = model_uri
        return b


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.BACKEND, GLUONCV_BACKEND,
                                            GluonCVBackendConfigBuilder)
