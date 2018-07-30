Based on https://github.com/AntreasAntoniou/DAGAN. This is an implementation of DAGAN as described in https://arxiv.org/abs/1711.04340.

Changes made include adding the bip dataset in `data.py` and the script to run an experiment.

# DAGAN
Implementation of DAGAN: Data Augmentation Generative Adversarial Networks

## Installation
Install python3 and pip3
```
sudo apt install -y python3 python3-pip
```

Then install dependencies

```
cd DAGAN/
pip install -r requirements.txt
```

## Datasets
Datasets should be placed in the following directories:
* /datasets/kont_data.npy
* /datasets/mani_data.npy
A symbolic link can be made
```
sudo ln -s /path/to/datasets/ /datasets
```

## Training a DAGAN

After the datasets are downloaded and the dependencies are installed, a DAGAN can be trained by running:

```
python3 train_bip_dagan.py --batch_size 10 --generator_inner_layers 2 --discriminator_inner_layers 3 --num_generations 64 --experiment_title bip_dagan_experiment --num_of_gpus 8 --z_dim 100 --dropout_rate_value 0.5 --continue_from_epoch 70
```

The last option may be removed in the first run.

## To Generate Data

The model training automatically uses unseen data to produce generations at the end of each epoch. However, once you have trained a model to satisfication you can generate samples for the whole of the validation set using the following command:

```
python3 gen_bip_dagan.py --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title bip_dagan_experiment --num_of_gpus 2 --z_dim 100 --dropout_rate_value 0.5
```
All the arguments must match the trained network's arguments and the `continue_from_epoch` argument must correspond to the epoch the trained model was at.
