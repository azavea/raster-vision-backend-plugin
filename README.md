# Raster Vision Backend Plugin Example

This repository holds an example of a new backend for Raster Vision, implemented as a plugin.

## Setup and Requirements

### Requirements

You'll need `docker` (preferably version 18 or above) installed.

### Setup

The Docker container in this repo inherits from `raster-vision-cpu` and `raster-vision-gpu`, so you'll need to have those built (using `./docker/build` in the `raster-vision` repo).

To build the image, run the following command:

```shell
> scripts/build
```

This will add some of this repo's code to the standard `raster-vision` image.

### Running the console

Whenever the instructions say to "run the console", it means to spin up an image and drop into a Bash shell by doing this:

```shell
> scripts/console
```

This will mount the following directories:
- `${HOME}/.aws` -> `/root/.aws`
- `${HOME}/.rastervision` -> `/root/.rastervision`
- `notebooks` -> `/opt/notebooks`
- `plugin` -> `/opt/src/plugin`
- `spacenet` -> `/opt/src/spacenet`
- `${RASTER_VISION_DATA_DIR}` -> `/opt/data`
- `${RASTER_VISION_REPO_DIR}/rastervision` -> `/opt/src/rastervision`

You will need to set the `RASTER_VISION_REPO_DIR` environment variable so it points to your local `raster-vision` repo. This will allow you to make changes to the `raster-vision` source code and have them immediately available in the running container.
You will also need to set `RASTER_VISION_DATA_DIR` to the directory containing your data.

### Running chip classification on Vegas

To drive development of a new chip classification backend, we will use the Vegas example in `spacenet/vegas.py`. It allows running all three tasks on the SpaceNet Vegas example. It is modified from the version in the examples repo to use a backend plugin (in `plugin/noop_backend.py`) when doing classification. See the `build_backend` method to see how this backend is utilized in the experiment configuration. You can run this example using something like the following. At the moment, it doesn't generate any real output because the backend is a noop. You may want to read about setting up the Vegas example [here](https://github.com/azavea/raster-vision-examples#spacenet-vegas-road-and-building-semantic-segmentation).
```
export ROOT_URI=/opt/data/lf-dev/rv-output/vegas-spacenet
rastervision -p plugin \
    run local -e spacenet.vegas \
    -a test True \
    -a use_remote_data False \
    -a root_uri ${ROOT_URI} \
    -a target buildings \
    -a task_type chip_classification
```
The `-p plugin` flag is to use a Raster Vision profile called `plugin`. This profile points to the location of the plugin module. You can make such a profile by creating a file at `~/.rastervision/plugin` containing something like:
```
[AWS_BATCH]
job_queue=raster-vision-gpu
job_definition=raster-vision-gpu
[PLUGINS]
files=[]
modules=["plugin.noop_backend"]
```
