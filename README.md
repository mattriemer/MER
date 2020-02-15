# Meta-Experience Replay (MER)

Source code for the paper "Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference".

Link: https://openreview.net/pdf?id=B1gTShAct7

Reference:
```
@inproceedings{MER,
    title={Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference},   
    author={Riemer, Matthew and Cases, Ignacio and Ajemian, Robert and Liu, Miao and Rish, Irina and Tu, Yuhai and Tesauro, Gerald},    
    booktitle={In International Conference on Learning Representations (ICLR)},    
    year={2019}   
}
```

This project is a fork of the GEM project https://github.com/facebookresearch/GradientEpisodicMemory in order to reproduce baselines from their paper. These baselines have been copied into the model/ directory of this repository. Our output and logging mechanisms for all models follow the same format used in the GEM project. 

# Available Datasets

The code in this repository should work on the variants of MNIST used in the experiments. This includes Rotations, Permutations, and Many Permutations. It would need to be slightly extended to be applied to other interesting continual learning benchmarks like CIFAR-100 or Omniglot.

The original MNIST database is available at http://yann.lecun.com/exdb/mnist/ and interface for generating your own MNIST variants is provided as part of the GEM project https://github.com/facebookresearch/GradientEpisodicMemory/tree/master/data. To maximize reproducibility, we have provided an interface for directly downloading the dataset versions used in our experiments.

## Basic Setup (Python 3.5 & torch 0.3.1)

As a first step to get up and running, clone this git repository and navigate into the root directory of your local version of the repository. To get started, please install the requirements inside your environment.

If you don't have an environment, we recommend that you create one (using [conda](http://anaconda.org)). The following instructions will guide you:

Install `conda` and type

```conda create --name mer python=3.5```

This will create a conda environment (an isolated workplace) in which we can install the right versions of the software. Then, activate the environment:

```source activate mer```

or

```conda activate mer```

Within the `mer` environment, install PyTorch and Cython using conda as follows:

```conda install pytorch=0.3.1 -c pytorch```

```conda install cython```

and then install the rest of the requirements using the following command:

```pip install --user -r requirements.txt```


## Basic Setup (Python 3.6+ & torch 1.4+)

As a first step to get up and running, clone this git repository and navigate into the root directory of your local version of the repository. To get started, please install the requirements inside your environment.

If you don't have an environment, we recommend that you create one (using [conda](http://anaconda.org)). The following instructions will guide you:

Install `conda` and type

```conda create --name mer python```

This will create a conda environment (an isolated workplace) in which we can install the right versions of the software. Then, activate the environment:

```source activate mer```

or

```conda activate mer```

Within the `mer` environment, install PyTorch and Cython using conda as follows:

```conda install torch```

```conda install cython```

_For python 3.6+, to install quadprog, first do:_
```sudo apt install gcc build-essential```

and then install the rest of the requirements using the following command:

```pip install --user -r requirements.txt```

## Getting the datasets

The first step is to download and uncompress all three datasets (30 GB of storage needed) execute the following command:

```python get_data.py all```

For just MNIST Rotations (4.1 GB) execute:

```python get_data.py rotations```

For just MNIST Permutations (4.1 GB) execute:

```python get_data.py permutations```

For just MNIST Many Permutations (21 GB) execute:

```python get_data.py manypermutations```

# Getting Started

In mer_examples.sh see examples of how to run variants of MER from the paper and baseline models from the experiments. We make sure first that this script is excutable:

```chmod +x mer_examples.sh```

Now, executing the following command leads to running a full suite of experiments with a random seed of 0:

```
./mer_examples.sh "0"
```

Within the file you can see examples of how to run models from our experiments on each datasets and memory size setting. For instance, we can execute MER from Algorithm 1 in the paper (meralg1) on MNIST Rotations with 5120 memories using the following commands:
```
export ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt --cuda no --seed 0"
python3 main.py $ROT --model meralg1 --lr 0.03 --beta 0.03 --gamma 1.0 --memories 5120 --replay_batch_size 100 --batches_per_example 10
```

# Available Models

In the model/ directory we have provided various models that were used for experiments on varients of MNIST in the paper. These models include:

- Online Learning (online)

- Task Specific Input Layer (taskinput)

- Independent Models per Task (independent)

- Elastic Weight Consolidation (ewc)

- Gradient Episodic Memory (gem)

- Experience Replay Algorithm 4 (eralg4)

- Experience Replay Algorithm 5 (eralg5)

- Meta-Experience Replay Algorithm 1 (meralg1)

- Meta-Experience Replay Algorithm 6 (meralg6)

- Meta-Experience Replay Algorithm 7 (meralg7)

# System Requirements

The repository has been developed for Python 3.5.2 using PyTorch 0.3.1.

# Reproducing Our Experiments 

We have conducted comprehensive experiments detailed in Appendix M of our paper to make sure that our results are reproducible across runs regardless of the machine and random seed. You should be able to reproduce these experiments using the provided mer_examples.sh script.
