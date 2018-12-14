import time
import logging

import numpy as np
import mxnet as mx

from mxnet import gluon, init, nd
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model

log = logging.getLogger(__name__)


def make_transform():
    """Define transform function for ImageFolderDataset in DataLoader.

    Args:
        none

    Returns:
        transform object for data loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform


def gpu_enabled():
    """Check if GPU is available.

    Args:
        none

    Returns:
        bool, True if GPU enabled, False if GPU not enabled
    """
    try:
        mx.nd.array([1, 2, 3], ctx=mx.gpu(0))
    except mx.MXNetError:
        return False
    return True


def to_mx(chip):
    """Convert numpy ndarray to mxnet array.

    Args:
        numpy ndarray

    Returns:
        mxnet ndarray
    """
    chip = nd.array(chip)
    return chip


def build_model(ctx, num_classes):
    """Define and initalize ResNet50v2 model.

    Args:
        ctx: (list) context, i.e. a compute environment
        num_classes: (int) number of classes

    Returns:
        initialized model

    """
    # GluonCV Model build here
    net = get_model('ResNet50_v2', pretrained=True)
    with net.name_scope():
        net.output = gluon.nn.Dense(num_classes)
    net.output.initialize(init.Xavier(), ctx=ctx)

    # set all layers to same context
    net.collect_params().reset_ctx(ctx)

    # hybridize, among other useful features, makes net serialization easier
    net.hybridize()

    return net


def test(model, ctx, val_data):
    """Run validation data through model and update val metric accordingly

    Args:
        model: gluoncv model
        ctx: (list) compute environment
        val_data: DataLoader containing validation data

    Returns:
        updated F1 score
    """
    metric = mx.metric.F1()

    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx,
                                          batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx,
                                           batch_axis=0)
        # Note: I am currently unsure why recording validation is necessary,
        # except that the F1 score is always 0 otherwise.
        with ag.record():
            outputs = [model(X) for X in data]
        metric.update(label, outputs)

    return metric.get()


def gluoncv_train(config_dict, model_path):
    """GluonCV Training Script, adapted from GluonCV classification examples:
        https://gluon-cv.mxnet.io/build/examples_classification/index.html

    Args:
        config_path: default dict containing relevant configuration
                     files, including classes and training_data_dir.
        model_path: path to which model is saved after each epoch

    Returns:
        trained model
    """

    epochs = 12
    per_device_batch_size = 32

    lr_decay = 0.1
    lr_decay_epoch = [10, 20, 30, 40, np.inf]
    num_devices = 1

    if gpu_enabled():
        ctx = [mx.gpu(i) for i in range(num_devices)]
    else:
        ctx = [mx.cpu(i) for i in range(num_devices)]
        log.info("GPU not enabled. Using CPU")

    batch_size = per_device_batch_size * max(num_devices, 1)
    optimizer = 'nag'

    optimizer_params = {'learning_rate': 0.01, 'wd': 0.0001, 'momentum': 0.9}

    model = build_model(ctx, config_dict['model']['nb_classes'])

    trainer = gluon.Trainer(model.collect_params(),
                            optimizer, optimizer_params)

    train_path = config_dict['trainer']['options']['training_data_dir']
    val_path = config_dict['trainer']['options']['validation_data_dir']

    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(train_path)
                         .transform_first(make_transform()),
        batch_size=batch_size, shuffle=True, last_batch='keep')
    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(val_path)
                         .transform_first(make_transform()),
        batch_size=batch_size, shuffle=False, last_batch='keep')

    train_metric = mx.metric.F1()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    num_batch = len(train_data)
    lr_decay_count = 0

    for epoch in range(epochs):
        tic = time.time()
        train_loss = 0
        train_metric.reset()

        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for batch in train_data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx,
                                              batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx,
                                               batch_axis=0)
            with ag.record():
                outputs = [model(X) for X in data]
                loss = [loss_fn(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            train_metric.update(label, outputs)

        _, train_f1 = train_metric.get()
        train_loss /= num_batch

        _, val_f1 = test(model, ctx, val_data)

        print('[Epoch %d] Train-f1: %.3f, loss: %.3f | '
              'Val-f1: %.3f | time: %.1f' %
              (epoch, train_f1, train_loss, val_f1, time.time() - tic))

        model.save_parameters(model_path)

    return model
