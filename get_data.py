# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import download
import argparse

def get_mnist_data(url, data_dir):
    print("Downloading {} into {}".format(url, data_dir))
    download.maybe_download_and_extract(url, data_dir)

def get_datasets():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Either the name of the dataset (rotations, permutations, manypermutations), or `all` to download all datasets")
    args = parser.parse_args()

    # Change dir to the location of this file (repo's root)
    get_data_path = os.path.realpath(__file__)
    os.chdir(os.path.dirname(get_data_path))
    data_dir = os.path.join(os.getcwd(), 'data')

    # get files
    mnist_rotations = "https://nlp.stanford.edu/data/mer/mnist_rotations.tar.gz"
    mnist_permutations = "https://nlp.stanford.edu/data/mer/mnist_permutations.tar.gz"
    mnist_many = "https://nlp.stanford.edu/data/mer/mnist_manypermutations.tar.gz"

    all = {"rotations": mnist_rotations, "permutations": mnist_permutations, "manypermutations": mnist_many}

    if args.dataset == "all":
        for dataset in all.values():
            get_mnist_data(dataset, data_dir)
    else:
        get_mnist_data(all[args.dataset], data_dir)

if __name__ == "__main__":
    get_datasets()
