# Udacity Self-Driving Car Engineer Nano Degree

## Ref
* https://www.youtube.com/watch?v=EaY5QiZwSP4
* https://github.com/llSourcell/How_to_simulate_a_self_driving_car
* https://github.com/udacity/self-driving-car-sim
* https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
* https://github.com/ndrplz/self-driving-car
* https://github.com/udacity?utf8=%E2%9C%93&q=car
* https://medium.com/deep-learning-turkey/behavioral-cloning-udacity-self-driving-car-project-generator-bottleneck-problem-in-using-gpu-182ee407dbc5

## Dependencies

You can install all dependencies by running one of the following commands

You need a [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to use the environment setting.

```python
# Use TensorFlow without GPU
conda env create -f udacity-environment.yml 

# Use TensorFlow with GPU
conda env create -f udacity-gpu-environment.yml
```

Or you can manually install the required libraries (see the contents of the environemnt*.yml files) using pip.


## Usage


### Run the pretrained model

Start up [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose a scene and press the Autonomous Mode button.  Then, run the model as follows:

```python
python drive.py model.h5
```

### To train the model

You'll need the data folder which contains the training images.

```python
python model.py
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.

## Credits

The credits for this code go to [naokishibuya](https://github.com/naokishibuya). I've merely created a wrapper to get people started.



