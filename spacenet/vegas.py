import re
import random
import os
from abc import abstractmethod

import rastervision as rv
from rastervision.utils.files import list_paths
from plugin import gluoncv_config

BUILDINGS = 'buildings'
ROADS = 'roads'


class SpacenetConfig(object):
    @staticmethod
    def create(use_remote_data, target):
        if target.lower() == BUILDINGS:
            return VegasBuildings(use_remote_data)
        elif target.lower() == ROADS:
            return VegasRoads(use_remote_data)
        else:
            raise ValueError('{} is not a valid target.'.format(target))

    def get_raster_source_uri(self, id):
        return os.path.join(
            self.base_uri, self.raster_dir,
            '{}{}.tif'.format(self.raster_fn_prefix, id))

    def get_label_source_uri(self, id):
        return os.path.join(
            self.base_uri, self.label_dir,
            '{}{}.geojson'.format(self.label_fn_prefix, id))

    def get_scene_ids(self):
        label_dir = os.path.join(self.base_uri, self.label_dir)
        label_paths = list_paths(label_dir, ext='.geojson')
        label_re = re.compile(r'.*{}(\d+)\.geojson'
                              .format(self.label_fn_prefix))
        scene_ids = [
            label_re.match(label_path).group(1)
            for label_path in label_paths]
        return scene_ids

    @abstractmethod
    def get_class_map(self):
        pass


class VegasRoads(SpacenetConfig):
    def __init__(self, use_remote_data):
        self.base_uri = '/opt/data/vegas-spacenet/AOI_2_Vegas_Roads_Train'
        if use_remote_data:
            self.base_uri = 's3://spacenet-dataset/SpaceNet_Roads_Competition/Train/AOI_2_Vegas_Roads_Train'  # noqa

        self.raster_dir = 'RGB-PanSharpen'
        self.label_dir = 'geojson/spacenetroads'
        self.raster_fn_prefix = 'RGB-PanSharpen_AOI_2_Vegas_img'
        self.label_fn_prefix = 'spacenetroads_AOI_2_Vegas_img'

    def get_class_map(self):
        # First class should be background when using GeoJSONRasterSource
        return {
            'Road': (1, 'orange'),
            'Background': (2, 'black')
        }


class VegasBuildings(SpacenetConfig):
    def __init__(self, use_remote_data):
        self.base_uri = '/opt/data/vegas-spacenet/AOI_2_Vegas_Train'
        if use_remote_data:
            self.base_uri = 's3://spacenet-dataset/SpaceNet_Buildings_Dataset_Round2/spacenetV2_Train/AOI_2_Vegas'  # noqa

        self.raster_dir = 'RGB-PanSharpen'
        self.label_dir = 'geojson/buildings'
        self.raster_fn_prefix = 'RGB-PanSharpen_AOI_2_Vegas_img'
        self.label_fn_prefix = 'buildings_AOI_2_Vegas_img'

    def get_class_map(self):
        # First class should be background when using GeoJSONRasterSource
        return {
            'Building': (1, 'orange'),
            'Background': (2, 'black')
        }


def build_scene(task, spacenet_config, id, channel_order=None):
    # Need to use stats_transformer because imagery is uint16.
    raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                      .with_uri(spacenet_config.get_raster_source_uri(id)) \
                      .with_channel_order(channel_order) \
                      .with_stats_transformer() \
                      .build()

    label_source_uri = spacenet_config.get_label_source_uri(id)

    if task.task_type == rv.SEMANTIC_SEGMENTATION:
        background_class_id = 2
        line_buffer = 15
        label_raster_source = rv.RasterSourceConfig.builder(rv.GEOJSON_SOURCE)\
            .with_uri(label_source_uri) \
            .with_rasterizer_options(background_class_id,
                                     line_buffer=line_buffer) \
            .build()

        label_source = rv.LabelSourceConfig \
            .builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
            .with_raster_source(label_raster_source) \
            .build()
    elif task.task_type == rv.CHIP_CLASSIFICATION:
        label_source = rv.LabelSourceConfig \
            .builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
            .with_uri(label_source_uri) \
            .with_ioa_thresh(0.01) \
            .with_use_intersection_over_cell(True) \
            .with_pick_min_class_id(True) \
            .with_background_class_id(2) \
            .with_infer_cells(True) \
            .build()
    elif task.task_type == rv.OBJECT_DETECTION:
        label_source = rv.LabelSourceConfig \
            .builder(rv.OBJECT_DETECTION_GEOJSON) \
            .with_uri(label_source_uri) \
            .build()

    scene = rv.SceneConfig.builder() \
                          .with_task(task) \
                          .with_id(id) \
                          .with_raster_source(raster_source) \
                          .with_label_source(label_source) \
                          .build()

    return scene


def build_dataset(task, spacenet_config, test):
    scene_ids = spacenet_config.get_scene_ids()
    if len(scene_ids) == 0:
        raise ValueError('No scenes found. '
                         'Something is configured incorrectly.')
    random.seed(5678)
    random.shuffle(scene_ids)
    split_ratio = 0.8
    num_train_ids = round(len(scene_ids) * split_ratio)
    train_ids = scene_ids[0:num_train_ids]
    val_ids = scene_ids[num_train_ids:]

    num_train_scenes = len(train_ids)-1
    num_val_scenes = len(val_ids)-1
    if test:
        num_train_scenes = 9
        num_val_scenes = 3
    train_ids = train_ids[0:num_train_scenes]
    val_ids = val_ids[0:num_val_scenes]
    channel_order = [0, 1, 2]

    train_scenes = [build_scene(task, spacenet_config, id, channel_order)
                    for id in train_ids]
    val_scenes = [build_scene(task, spacenet_config, id, channel_order)
                  for id in val_ids]
    dataset = rv.DatasetConfig.builder() \
                              .with_train_scenes(train_scenes) \
                              .with_validation_scenes(val_scenes) \
                              .build()
    return dataset


def build_task(task_type, class_map):
    if task_type == rv.SEMANTIC_SEGMENTATION:
        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(300) \
                            .with_classes(class_map) \
                            .with_chip_options(
                                chips_per_scene=9,
                                debug_chip_probability=1.0,
                                negative_survival_probability=0.25,
                                target_classes=[1],
                                target_count_threshold=1000) \
                            .build()
    elif task_type == rv.CHIP_CLASSIFICATION:
        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                    .with_chip_size(300) \
                    .with_classes(class_map) \
                    .build()
    elif task_type == rv.OBJECT_DETECTION:
        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes(class_map) \
                            .with_chip_options(neg_ratio=1.0,
                                               ioa_thresh=0.8) \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

    return task


def build_backend(task, test):
    debug = False
    if test:
        debug = True

    if task.task_type == rv.SEMANTIC_SEGMENTATION:
        batch_size = 8
        num_steps = 1e6
        if test:
            num_steps = 16
            batch_size = 4

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(rv.MOBILENET_V2) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(debug) \
                                  .build()
    elif task.task_type == rv.CHIP_CLASSIFICATION:
        backend = rv.BackendConfig.builder(gluoncv_config.GLUONCV_BACKEND) \
                                  .with_task(task) \
                                  .build()
    elif task.task_type == rv.OBJECT_DETECTION:
        batch_size = 8
        num_steps = 1e6
        if test:
            num_steps = 16
            batch_size = 4

        backend = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
            .with_task(task) \
            .with_model_defaults(rv.SSD_MOBILENET_V1_COCO) \
            .with_debug(debug) \
            .with_batch_size(batch_size) \
            .with_num_steps(num_steps)

    return backend


def str_to_bool(x):
    if type(x) == str:
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
        else:
            raise ValueError('{} is expected to be true or false'.format(x))
    return x


def validate_options(task_type, target):
    if task_type not in [rv.SEMANTIC_SEGMENTATION, rv.CHIP_CLASSIFICATION,
                         rv.OBJECT_DETECTION]:
        raise ValueError('{} is not a valid task_type'.format(task_type))

    if target not in [ROADS, BUILDINGS]:
        raise ValueError('{} is not a valid target'.format(target))

    if target == ROADS:
        if task_type in [rv.CHIP_CLASSIFICATION, rv.OBJECT_DETECTION]:
            raise ValueError('{} is not valid task_type '
                             'for target="roads"'.format(task_type))


class SpacenetVegas(rv.ExperimentSet):
    def exp_main(self, root_uri, target=BUILDINGS, use_remote_data=True,
                 test=False, task_type=rv.CHIP_CLASSIFICATION):
        """Run an experiment on the Spacenet Vegas road or building dataset.

        This is an example of how to do all three tasks on the same dataset.

        Args:
            root_uri: (str): root of where to put output
            target: (str) 'buildings' or 'roads'
            use_remote_data: (bool or str) if True or 'True', then use data
                from S3, else local
            test: (bool or str) if True or 'True', run a very small experiment
                as a test and generate debug output
            task_type: (str) valid options are semantic_segmentation,
                object_detection, and chip_classification
        """
        test = str_to_bool(test)
        task_type = task_type.upper()
        use_remote_data = str_to_bool(use_remote_data)
        spacenet_config = SpacenetConfig.create(use_remote_data, target)
        experiment_id = '{}_{}'.format(target, task_type.lower())
        validate_options(task_type, target)
        task = build_task(task_type, spacenet_config.get_class_map())
        backend = build_backend(task, test)
        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                    .build()
        dataset = build_dataset(task, spacenet_config, test)
        # Need to use stats_analyzer because imagery is uint16.
        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(experiment_id) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_analyzer(analyzer) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
