#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainer.computational_graph as c
import numpy as np
import requests as rq
from ourfunctions import *
import datetime
import json
import os,stat
import platform


def main():
    if platform.system()=='Windows':
        import colorama
        colorama.init()
        path=os.getcwd()+"\\files4runtime\\"
    if platform.system()=='Linux':  #aka android
        path=r"storage/emulated/0/Download/"    #TODO - maybe need to create the MNISTDist directory
    '''parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--IP', '-i', type=str, default="127.0.0.1",
                        help='server IP address')
    parser.add_argument('--port', '-p', type=str, default="8000",
                        help='server port')'''
    print("hi\n"+path+"\nbye")
    try:
        os.makedirs(path)
    except:
        pass
        #do nothing

    url= r"http://127.0.0.1:8000/MNIST/"
    ####################################################################
    #imalive
    ####################################################################
    device_model="bar computer"		#TODO - acually getting device model, i.e Sony Xperia Z3 compact
    result=rq.post(url+"imalive",data=device_model)
    deviceId = result.text.split()[1]#TODO - more robust parsing
    dataSetUrl=result.text.split()[3]#TODO - more robust parsing


    #create datasets directory
    try:
        os.makedirs(path+r'pfnet\chainer\mnist')
    except:
        pass
    ####################################################################
	#getTrainSet
    #####################################################################
    if not os.path.isfile(path + r'pfnet\chainer\mnist' + r"\train.npz"):
        result = rq.post(url + "getTrainSet", data=device_model)
        with open(path+r'pfnet\chainer\mnist' + r"\train.npz", 'wb') as fd:
            for chunk in result.iter_content(10):
                fd.write(chunk)

    ####################################################################
    # getTestSet
    ####################################################################
    if not os.path.isfile(path+r'pfnet\chainer\mnist' + r"\test.npz"):
        result = rq.post(url + "getTestSet", data=device_model)
        with open(path+r'pfnet\chainer\mnist' + r"\test.npz", 'wb') as fd:
            for chunk in result.iter_content(10):
                fd.write(chunk)

    chainer.dataset.set_dataset_root(path)
    train, test = downloadData(dataSetUrl)


    #TODO - some kind of a loop
    while(True):

        ####################################################################
        #getNeuralNet
        ####################################################################
        result=rq.post(url+"getNeuralNet",data=device_model)
        with open(path+"getNeuralNet.npz", 'wb') as fd:
            for chunk in result.iter_content(10):
                fd.write(chunk)
        NeuralNet=L.Classifier(MLP(784, 10, 10))
        chainer.serializers.load_npz(path+"getNeuralNet.npz",NeuralNet)

        ####################################################################
        #getData
        ####################################################################
        result=rq.post(url+"getData",data=deviceId)
        with open(path+"getData.npz", 'wb') as fd:
            for chunk in result.iter_content(10):
                fd.write(chunk)
        temp=np.load(path+"getData.npz")
        isTrain=				temp['isTrain']
        minibatchID=			temp['minibatchID']
        epochNumber=			temp['epochNumber']
        subsetDataForDevice=	temp['subsetDataForDevice']

        computSet=extractInputs(train,subsetDataForDevice)
        ''' g = c.build_computational_graph((model.loss,))
        with open('C:\mynet', 'w') as o:
            o.write(g.dump())'''

        #computing
        if isTrain:
            originalNeuralNet=NeuralNet
            trainedNeuralNet=deviceTrain(NeuralNet,computSet)
            #trainedNeuralNet=np.load(trainedNeuralNetPath)
            computedResult=calcDelta(originalNeuralNet,trainedNeuralNet)
        else :

            computedResult=deviceValidate(NeuralNet,computSet)

        ####################################################################
        #postData
        ####################################################################

        #np.savez(path+"postData", deviceID=deviceId,epochNumber=epochNumber,computingTime="not now",computedResult=computedResult)
        data={ 'deviceId':str(deviceId),'miniBatchID':str(minibatchID), 'epochNumber':str(epochNumber), 'computingTime':"not now",'computedResult':computedResult}
        #np.savez(path+"postData", deviceId=str(deviceId), epochNumber=str(epochNumber), computingTime="not now",computedResult=computedResult)
        rq.post(url + "postData", data=json.dumps(data))
        #response = FileResponse(open(path+"postData", 'rb'))
    #response['Content-Disposition'] = 'attachment; filename=Data.npz'
    return

if __name__ == '__main__':
    main()
