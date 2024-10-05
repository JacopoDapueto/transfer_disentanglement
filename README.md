# Transferring disentangled representations: bridging the gap between synthetic and real images
[[Paper](https://arxiv.org/abs/2409.18017)]

---

## 🔧 Dependencies and Installation
### Install dependencies
The code was developed and tested with Python 3.10 and the main dependencies are [Pytorch == 2.0.0 ](https://pytorch.org/) and [Cuda==11.7](https://developer.nvidia.com/cuda-toolkit)

Set up the environment and then install the dependencies with
```
pip install -r requirements.txt
```

### Download and prepare datasets

1. Set the environment variable `DISENTANGLEMENT_TRANSFER_DATA` to this path, for example by adding

```
export DISENTANGLEMENT_TRANSFER_DATA=<path to the data directory>
```
2. Download all the necessary datasets with the script
```
./bash_scripts/download_datasets.sh
```
3. Unzip the compressed files (Coil100 and RGDB Objects)

4. Create the augmented and binary version of Coil100 with the command
```
python create_coil100_augmented/augment_coil100.py
```
## OMES usage
The code requires `representations.npz` and `classes.csv` to contain the representation and the labels of the FoVs of random samples. Both files are in the same directory <path to representation>

```
python compute_omes.py --representation_directory <path to representation>
```


## How to reproduce Transfer experiments

To reproduce the experiment of the study use the scripts in the folder 
```
./bash_scripts
```

### Train Source models
The scripts starting with *train_* execute the training of the Source models.

```
bash ./bash_scripts/train_*.sh
```

The results will be saved in `./outuput` directory, organized by _experiment name_ and numbered by the _random seed_.

**Once** one experiment folder is completed aggregate the results of all random seeds with the scripts
```
python dlib_aggregate_results_experiment.py --experiment experiment_name 
```


### Transfer on Target dataset
**Once** you have trained the source models, run the scripts to execute the transfer experiments.

```
bash ./bash_scripts/transfer_*.sh
```

The results will be saved in `./outuput` directory, organized by experiment name:
*source_experiment*\_to\_*target_experiment*

**Once** one experiment folder is completed aggregate the results of all random seeds with the scripts
```
python dlib_aggregate_results_transfer_experiment.py --experiment experiment_name 
```
**Then** to read and save the results of the experiment run
```
python dlib_compare_transfer.py --experiment experiment_name --values_to_aggregate "model_num"
```

## Contacts
If you have any questions, you are very welcome to email jacopo.dapueto@gmail.com

## Bibtex
If you find this useful for your research and applications, please cite using this BibTeX:

```BibTeX
@article{dapueto2024transferring,
  title={Transferring disentangled representations: bridging the gap between synthetic and real images},
  author={Dapueto, Jacopo and Noceti, Nicoletta and Odone, Francesca},
  journal={arXiv preprint arXiv:2409.18017},
  year={2024}
}
