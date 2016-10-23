from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import colorama
colorama.init()
import chainer.computational_graph as c
import numpy as np
import requests as rq

path = "C:\\temp\\"  # TODO - need to change to android path


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def extractInputs(dataset,indices):
    '''
    extracts specified inputs from dataset
    :param dataset: dataset to be spliced
    :param indices: desired indices to comput
    :return: spliced dataset
    '''
    sub1,sub2=chainer.datasets.split_dataset(dataset,500)
    return sub1

def downloadData(dataSetUrl):
    '''

    :param dataSetUrl: url to download dataset from
    :return: return trainset, testset
    '''
    train, test=chainer.datasets.get_mnist()  # TODO - using urls from our server. maybe testing how long it actually takes to download from original site
    return train, test

def deviceTrain(NeuralNet,computSet):
    '''
    :param NeuralNet: neural network to train
    :param computSet: dataset to train on
    :return: trained neural network model
    '''
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(NeuralNet)

    # Load the MNIST dataset
    train_iter = chainer.iterators.SerialIterator(computSet, len(
        computSet[1]))  # TODO - make sure that len(computSet)=len(subsetDataForDevice)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (1, 'epoch'), out=path + "trainingResult.npz")

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # chainer.serializers.load_npz(True, trainer)

    # Run the training
    #trainer.extend(extensions.snapshot(filename=path+"trainedNeuralNet"))
    chainer.serializers.save_npz(path+'pre',NeuralNet)
    trainer.run()
    chainer.serializers.save_npz(path+'post',NeuralNet)

    return NeuralNet

def deviceValidate(NeuralNet, computSet):
    '''

    :param NeuralNet: neural network to validate
    :param computSet: dataset to validate with
    :return: accuracy in percent
    '''
    return 0.5

def calcDelta(originalNeuralNet,trainedNeuralNet):
    '''

    :param originalNeuralNet: neural network before training
    :param trainedNeuralNet: neural network after training
    :return: basically elementwise trainedNeuralNet - originalNeuralNet
    '''
    #chainer.serializers.save_npz(path+"originalNeuralNet",originalNeuralNet)
    #chainer.serializers.save_npz(path+"trainedNeuralNet",trainedNeuralNet)
    o=np.load(path+"pre")
    t=np.load(path+"post")
    return t['predictor/l3/W']-o['predictor/l3/W']  #TODO - of course this is not all
