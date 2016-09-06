
from parameters import DATASET, NETWORK
import numpy as np

def load_data(validation=False, test=False):
    
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    if DATASET.name == "Fer2013":

        # load images
        data_dict['X'] = np.load(DATASET.train_folder + '/images.npy')
        if DATASET.trunc_trainset_to > 0:
            data_dict['X'] = data_dict['X'][0:DATASET.trunc_trainset_to, :, :]
        data_dict['X'] = data_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        # load landmarks
        data_dict['X2'] = np.load(DATASET.train_folder + '/landmarks.npy')
        if DATASET.trunc_trainset_to > 0:
            data_dict['X2'] = data_dict['X2'][0:DATASET.trunc_trainset_to, :, :]
        # load labels
        data_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')

        ###########################################
        ## TODO : load landmarks in validation and test set also. I'm too sleepy to do it today -_- zzz... zzz...
        ###########################################
        validation_dict['X'] = data_dict['X'][:500, :, :, :]
        validation_dict['X2'] = data_dict['X2'][0:500, :, :]
        validation_dict['Y'] = data_dict['Y'][0:500, :]
        return data_dict, validation_dict 
        ###########################################

        if validation:
            validation_dict['X'] = np.load(DATASET.validation_folder + '/images.npy')
            if DATASET.trunc_validationset_to > 0:
                validation_dict['X'] = validation_dict['X'][0:DATASET.trunc_validationset_to, :, :]
            validation_dict['X'] = validation_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')
        
        if test:
            test_dict['X'] = np.load(DATASET.test_folder + '/images.npy')
            if DATASET.trunc_testset_to > 0:
                test_dict['X'] = test_dict['X'][0:DATASET.trunc_testset_to, :, :]
            test_dict['X'] = test_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            test_dict['Y'] = np.load(DATASET.test_folder + '/labels.npy')

        if not validation and not test:
            return data_dict
        elif not test:
            return data_dict, validation_dict
        else: 
            return data_dict, validation_dict, test_dict
    else:
        print "Inknown dataset"
        exit()