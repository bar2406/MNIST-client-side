from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import platform
import numpy as np
import os

if platform.system() == 'Windows':  # for testing and debug on pc, path is different from android
    import colorama

    colorama.init()
    path = os.getcwd() + "\\files4runtime\\"
if platform.system() == 'Linux':  # aka android
    path = r"storage/emulated/0/Download/files4runtime/"  # any directory with r\w permissions will do


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
    order = list(range(dataset._length))
    temp = sorted(indices)
    for i in temp[::-1]:
        del order[i]
    order = list(indices) + order
    sub1 = chainer.datasets.SubDataset(dataset, 0, len(indices), order=order)
    return sub1


def downloadData(dataSetUrl):
    '''

    :param dataSetUrl: url to download dataset from
    :return: return trainset, testset
    '''

    train, test = chainer.datasets.get_mnist()
    return train, test


def deviceTrain(NeuralNet, computSet):
    '''
    :param NeuralNet: neural network to train
    :param computSet: dataset to train on
    :return: trained neural network model
    '''
	##changable constans:
    LEARN_RATE = 0.01
    optimizer = chainer.optimizers.SGD(LEARN_RATE) ##Also the optimizer is changable
    LOCAL_BATCH_SIZE = 100
	
	####################################################################
	
	
    # Setup an optimizer
    # optimizer = chainer.optimizers.SGD()
    optimizer.setup(NeuralNet)

    # Load the MNIST dataset
    train_iter = chainer.iterators.SerialIterator(computSet, LOCAL_BATCH_SIZE, repeat=False, shuffle=False)

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
    chainer.serializers.save_npz(path + 'pre', NeuralNet)
    trainer.run()
    chainer.serializers.save_npz(path + 'post', NeuralNet)


    return NeuralNet


def deviceValidate(NeuralNet, computSet):
    '''

    :param NeuralNet: neural network to validate
    :param computSet: dataset to validate with
    :return: accuracy in percent
    '''
    test_iter = chainer.iterators.SerialIterator(computSet, computSet._size, repeat=False)
    eval = extensions.Evaluator(test_iter, NeuralNet)
    return eval()['main/accuracy']


def calcDelta(originalNeuralNet, trainedNeuralNet):
    '''

    :param originalNeuralNet: neural network before training
    :param trainedNeuralNet: neural network after training
    :return: basically elementwise trainedNeuralNet - originalNeuralNet
    '''
    o = np.load(path + "pre")
    t = np.load(path + "post")
    delta = dict(t)

    for f in o.files:
        delta[f]=t[f]-o[f]
        delta[f]=delta[f].tolist()

    return delta
