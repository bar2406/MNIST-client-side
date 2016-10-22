#!/usr/bin/env python
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
# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_units),  # second layer
            l4=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
	path=r"C:\temp\"	#TODO - need to change to android path
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--IP', '-i', type=str, default="127.0.0.1",
                        help='server IP address')
    parser.add_argument('--port', '-p', type=str, default="8000",
                        help='server port')
	url= r"http://" + args.IP + ":" + args.port + r"/MNIST/"
	
	#imalive
	device_model="my model"		#TODO - acually getting device model, i.e Sony Xperia Z3 compact
	result=rq.post(url+"imalive",data=device_model)
	deviceId = p.text.split()[1]#TODO - more robust parsing 	
	dataSetUrl=p.text.split()[3]#TODO - more robust parsing 	
	
	#downloading dataset
	train, test = chainer.datasets.get_mnist()	#TODO - using urls from our server. maybe testing how long it actually takes to download from original site
	
	#some kind of a loop
	#getData
	
	result=rq.post(url+"getData",data=deviceId)
	with open(path+"getData.npz", 'wb') as fd:
	for chunk in result.iter_content(10):
		fd.write(chunk)
	temp=np.load(path+"getData.npz")
	
	isTrain=				temp['isTrain']
	minibatchID=			temp['minibatchID']
	epochNumber=			temp['epochNumber']
	subsetDataForDevice=	temp['subsetDataForDevice']
	neuralNet=				temp['neuralNet']	#expecting neuralNet to be a snapshot of the network, i.e of class model
	
   ''' g = c.build_computational_graph((model.loss,))
    with open('C:\mynet', 'w') as o:
        o.write(g.dump())'''
	
	#computing
	
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(neuralNet)

    # Load the MNIST dataset
    train1, train2 = chainer.datasets.split_dataset(subsetDataForDevice)
    print("train1 size:" + str(train1._size))
    print("train2 size:" + str(train2._size))
    train_iter = chainer.iterators.SerialIterator(train1, len(subsetDataForDevice))


    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (1, 'epoch'), out=path+"trainingResult.npz")

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
	
	chainer.serializers.load_npz(True, trainer) #TODO - perhaps should be False

    # Run the training
    trainer.run()
    
	#extracting results
	chainer.serializers.save_npz(path+'final_npz',neuralNet)
	
	
	
	
if __name__ == '__main__':
    main()
