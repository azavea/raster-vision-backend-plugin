from os.path import join
import os
import shutil
import uuid

import numpy as np
import mxnet as mx

from mxnet import gluon

from rastervision.backend import Backend
from rastervision.utils.files import (make_dir, get_local_path, upload_or_copy,
                                      download_if_needed)
from rastervision.utils.misc import save_img
from rastervision.data import ChipClassificationLabels

from plugin.train import (build_model, gpu_enabled, gluoncv_train)


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

    def get_backend_config(self, pretrained_model_uri, gcv_config,
                           dataset_files, class_map):

        from collections import defaultdict
        config = defaultdict(lambda: defaultdict(dict))

        # Update config using local paths.
        config['trainer']['options']['output_dir'] = \
            self.get_local_path(self.base_uri)
        config['model']['model_path'] = self.get_local_path(self.model_uri)
        config['model']['nb_classes'] = len(class_map)
        config['trainer']['options']['training_data_dir'] = \
            dataset_files.get_local_path(dataset_files.training_uri)
        config['trainer']['options']['validation_data_dir'] = \
            dataset_files.get_local_path(dataset_files.validation_uri)
        config['trainer']['options']['class_names'] = \
            class_map.get_class_names()

        # Save the pretrained weights locally
        pretrained_model_path = None
        if pretrained_model_uri:
            pretrained_model_path = self.download_if_needed(
                pretrained_model_uri)

        return (config, pretrained_model_path)


class GluonCVBackend(Backend):
    def __init__(self, backend_config, task_config):
        self.model = None
        self.model_path = None
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
            if class_id is None:
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
        """Find dataset and model locations for use in training script. Then,
           update model files with trained model.

        Args:
            tmp_dir: temporary directory

        Returns:
            None
        """
        dataset_files = DatasetFiles(self.config.training_data_uri, tmp_dir)
        dataset_files.download()

        model_files = ModelFiles(
            self.config.training_output_uri,
            tmp_dir)
        model_paths = model_files.get_backend_config(
            self.config.pretrained_model_uri, self.config.gcv_config,
            dataset_files, self.class_map)

        backend_config_dict, pretrained_model_path = model_paths
        self.model_path = backend_config_dict['model']['model_path']

        self.model = gluoncv_train(backend_config_dict, self.model_path)

        upload_or_copy(self.model_path, model_files.model_uri)

    def load_model(self, tmp_dir, ctx=None):
        """Load ResNet50v2 and load trained weights.

        Args:
            tmp_dir: temporary directory

        Returns:
            None
        """
        if self.model is None:
            self.model = build_model(ctx, len(self.class_map))
            model_files = ModelFiles(self.config.training_output_uri, tmp_dir)
            model_path = download_if_needed(model_files.model_uri, tmp_dir)
            self.model.load_parameters(model_path)

    def predict(self, chips, windows, tmp_dir):
        """Predict on a set of chips and return a ChipClassificationLabels
           object with class and probabilities.

        Args:
            chips: 300x300 jpgs
            windows: list of int
            tmp_dir: temporary directory

        Returns:
            labels and probabilities
        """
        # set compute environment depending on gpu availability
        ctx = [mx.gpu(0)] if gpu_enabled() else [mx.cpu(0)]

        # ensure model is loaded
        self.load_model(tmp_dir, ctx)

        # reshape the data to be consistent with how the model was trained
        data = np.transpose(chips, (0, 3, 2, 1))

        # load data onto gpu/cpu, depending on context
        data = gluon.utils.split_and_load(data, ctx)

        # get list of predictions from the model. also convert to np ndarray
        # because that is what raster vision expects
        probs = [self.model(X).asnumpy() for X in data]

        labels = ChipClassificationLabels()
        for chip_probs, window in zip(probs[0], windows):
            # Add 1 to class_id since they start at 1.
            class_id = int(np.argmax(chip_probs) + 1)
            labels.set_cell(window, class_id, chip_probs)

        return labels
