#!/usr/bin/env python
from __future__ import print_function

import requests as rq
from ourfunctions import *
import json
import os
import platform
import datetime

def main():
	##changable constans:
    MIDDLE_LAYER_SIZE = 100
	
	
	####################################################################

    if platform.system()=='Windows':    #for testing and debug on pc, path is different from android
        import colorama
        colorama.init()
        path=os.getcwd()+"\\files4runtime\\"
    if platform.system()=='Linux':  #aka android
        path=r"storage/emulated/0/Download/files4runtime/"  #any directory with r\w permissions will do

    dataset_root=os.path.join(path,"pfnet","chainer","mnist")
    #creating directory
    if not os.path.exists(path):
        try:
            original_umask = os.umask(0)
            os.makedirs(path)
        finally:
            os.umask(original_umask)

    url= r"http://192.168.1.8:8123/MNIST/"
    ####################################################################
    #imalive
    ####################################################################
    device_model="Galaxy j716"		#TODO -change for each device (any alias name, doesn't have to be unique)
    result=rq.post(url+"imalive",data=device_model)
    if result.status_code!=200 :
        raise RuntimeError("error 47: response code isn't 200")
    deviceId = result.text.split()[1]
    dataSetUrl=None #result.text.split()[3] - if we want to download from somewhere else, right now it's not in use


    #create datasets directory
    if not os.path.exists(dataset_root):
        try:
            original_umask = os.umask(0)
            os.makedirs(dataset_root)
        finally:
            os.umask(original_umask)

    ####################################################################
	#getTrainSet
    #####################################################################
    if not os.path.isfile(os.path.join(dataset_root,"train.npz")):  #if we didn't already downloaded the dataset
        result = rq.post(url + "getTrainSet", data=device_model)
        if result.status_code != 200:
            raise RuntimeError("error 63: response code isn't 200")
        with open(os.path.join(dataset_root,"train.npz"), 'wb') as fd:
            for chunk in result.iter_content(10):
                fd.write(chunk)

    ####################################################################
    # getTestSet
    ####################################################################
    if not os.path.isfile(os.path.join(dataset_root,"test.npz")):   #if we didn't already downloaded the dataset
        result = rq.post(url + "getTestSet", data=device_model)
        if result.status_code != 200:
            raise RuntimeError("error 74: response code isn't 200")
        with open(os.path.join(dataset_root,"test.npz"), 'wb') as fd:
            for chunk in result.iter_content(10):
                fd.write(chunk)

    chainer.dataset.set_dataset_root(path)  #point chainer to the correct dataset root
    train, test = downloadData(dataSetUrl)


    while (True): #Runs until the server says no more missions

        ####################################################################
        #getNeuralNet
        ####################################################################
        time_delta=datetime.datetime.now()
        result=rq.post(url+"getNeuralNet",data=device_model)
        if result.status_code != 200:
            raise RuntimeError("error 91: response code isn't 200")
        with open(os.path.join(path,"getNeuralNet.npz"), 'wb') as fd:
            for chunk in result.iter_content(10):
                fd.write(chunk)
        NeuralNet=L.Classifier(MLP(784, MIDDLE_LAYER_SIZE, 10))   #network size must be the same as defined in the server TODO - maybe get net size from server
        chainer.serializers.load_npz(path+"getNeuralNet.npz",NeuralNet)
        time_delta=datetime.datetime.now()-time_delta
        print("getNeuralNet time:   "+str(time_delta.total_seconds()))
        ####################################################################
        #getData
        ####################################################################
        time_delta=datetime.datetime.now()
        result=rq.post(url+"getData",data=deviceId)
        if result.status_code != 200:
            raise RuntimeError("error 103: response code isn't 200")
        with open(os.path.join(path,"getData.npz"), 'wb') as fd:
            for chunk in result.iter_content(10):
                fd.write(chunk)
        time_delta=datetime.datetime.now()-time_delta
        print("getData time:        "+str(time_delta.total_seconds()))
        temp=np.load(path+"getData.npz")
        isTrain=				temp['isTrain']
        minibatchID=			temp['minibatchID']
        epochNumber=			temp['epochNumber']
        subsetDataForDevice=	temp['subsetDataForDevice']
        isTestset=              temp['isTestset']
        isFinished=             temp['isFinished']

        if isFinished:
            print ("No more missions from the server, bye bye")
            return #Finished to run, stop program

        computSet=extractInputs(train,subsetDataForDevice)
        computingTime=datetime.datetime.now()
        if isTestset: #Testing the network
            computSet=extractInputs(test,subsetDataForDevice)
            computedResult=deviceValidate(NeuralNet,computSet)
            computingTime = datetime.datetime.now() - computingTime
            print("computing test time:    " + str(computingTime.total_seconds()))
        else:
            if isTrain: #Training the network
                #NeuralNet.cleargrads()
                originalNeuralNet=NeuralNet
                trainedNeuralNet=deviceTrain(NeuralNet,computSet)
                computedResult=calcDelta(originalNeuralNet,trainedNeuralNet)
                computingTime = datetime.datetime.now() - computingTime
                print("computing train time:    " + str(computingTime.total_seconds()))
            else : #Validating the network
                computedResult=deviceValidate(NeuralNet,computSet)
                computingTime=datetime.datetime.now()-computingTime
                print("computing test time:     "+str(computingTime.total_seconds()))



        ####################################################################
        #postData
        ####################################################################
        time_delta=datetime.datetime.now()
        data={ 'deviceId':str(deviceId),'miniBatchID':str(minibatchID), 'epochNumber':str(epochNumber), 'computingTime':str(computingTime.total_seconds()),'computedResult':computedResult,'accuracy':str(deviceValidate(NeuralNet, computSet))} #TODO - maybe send computing time
        rq.post(url + "postData", data=json.dumps(data))
        time_delta=datetime.datetime.now()-time_delta
        print("postData time:        "+str(time_delta.total_seconds()))

    return

if __name__ == '__main__':
    main()
