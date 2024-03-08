# Transfer disentanglement

## Install

The code was developed and tested with Python 3.10 and the main dependences are Pytorch 2.0.0 and Cuda 11.7

Set up the environment and then install the dependencies with
```
pip install -r requirements.txt
```

Set the environment variable `DISENTANGLEMENT_TRANSFER_DATA` to this path, for example by adding

```
export DISENTANGLEMENT_TRANSFER_DATA=<path to the data directory>
```

## How to reproduce 

To reproduce the experiment of the study use the scripts in folder 
```
./bash_scripts
```

### Train Source models
The scripts starting with *train_* execute the training of the Source models.

```
bash ./bash_scripts/train_*.sh
```

The results will be save in `./outuput` directory, organized by experiment name.


### Transfer on Target dataset
Once you trained the source models, run the scripts to execute the transfer experiments.

```
bash ./bash_scripts/transfer_*.sh
```

The results will be save in `./outuput` directory, organized by experiment name:
*source_experiment*_to_*target_experiment*



