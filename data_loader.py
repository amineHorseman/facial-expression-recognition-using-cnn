"""
@AmineHorseman
Sep, 1st, 2016
"""
from parameters import DATASET, NETWORK
import numpy as np

def load_data(validation=False, test=False):
    
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    if DATASET.name == "Fer2013":

        # load train set
        data_dict['X'] = np.load(DATASET.train_folder + '/images.npy')
        data_dict['X'] = data_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        if NETWORK.use_landmarks:
            data_dict['X2'] = np.load(DATASET.train_folder + '/landmarks.npy')
        if NETWORK.use_hog_and_landmarks:
            data_dict['X2'] = np.load(DATASET.train_folder + '/landmarks.npy')
            data_dict['X2'] = np.array([x.flatten() for x in data_dict['X2']])
            data_dict['X2'] = np.concatenate((data_dict['X2'], np.load(DATASET.train_folder + '/hog_features.npy')), axis=1)
        data_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')
        if DATASET.trunc_trainset_to > 0:
            data_dict['X'] = data_dict['X'][0:DATASET.trunc_trainset_to, :, :]
            if NETWORK.use_landmarks and NETWORK.use_hog_and_landmarks:
                data_dict['X2'] = data_dict['X2'][0:DATASET.trunc_trainset_to, :]
            elif NETWORK.use_landmarks:
                data_dict['X2'] = data_dict['X2'][0:DATASET.trunc_trainset_to, :, :]
            data_dict['Y'] = data_dict['Y'][0:DATASET.trunc_trainset_to, :]

        if validation:
            # load validation set
            validation_dict['X'] = np.load(DATASET.validation_folder + '/images.npy')
            validation_dict['X'] = validation_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            if NETWORK.use_landmarks:
                validation_dict['X2'] = np.load(DATASET.validation_folder + '/landmarks.npy')
            if NETWORK.use_hog_and_landmarks:
                validation_dict['X2'] = np.load(DATASET.validation_folder + '/landmarks.npy')
                validation_dict['X2'] = np.array([x.flatten() for x in validation_dict['X2']])
                validation_dict['X2'] = np.concatenate((validation_dict['X2'], np.load(DATASET.validation_folder + '/hog_features.npy')), axis=1)
            validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')
            if DATASET.trunc_validationset_to > 0:
                validation_dict['X'] = validation_dict['X'][0:DATASET.trunc_validationset_to, :, :]
                if NETWORK.use_landmarks and NETWORK.use_hog_and_landmarks:
                    validation_dict['X2'] = validation_dict['X2'][0:DATASET.trunc_validationset_to, :]
                elif NETWORK.use_landmarks:
                    validation_dict['X2'] = validation_dict['X2'][0:DATASET.trunc_validationset_to, :, :]
                validation_dict['Y'] = validation_dict['Y'][0:DATASET.trunc_validationset_to, :]
        
        if test:
            # load test set
            test_dict['X'] = np.load(DATASET.test_folder + '/images.npy')
            test_dict['X'] = test_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            if NETWORK.use_landmarks:
                test_dict['X2'] = np.load(DATASET.test_folder + '/landmarks.npy')
            if NETWORK.use_hog_and_landmarks:
                test_dict['X2'] = np.load(DATASET.test_folder + '/landmarks.npy')
                test_dict['X2'] = np.array([x.flatten() for x in test_dict['X2']])
                test_dict['X2'] = np.concatenate((test_dict['X2'], np.load(DATASET.test_folder + '/hog_features.npy')), axis=1)
            test_dict['Y'] = np.load(DATASET.test_folder + '/labels.npy')
            if DATASET.trunc_testset_to > 0:
                test_dict['X'] = test_dict['X'][0:DATASET.trunc_testset_to, :, :]
                if NETWORK.use_landmarks and NETWORK.use_hog_and_landmarks:
                    test_dict['X2'] = test_dict['X2'][0:DATASET.trunc_testset_to, :]
                elif NETWORK.use_landmarks:
                    test_dict['X2'] = test_dict['X2'][0:DATASET.trunc_testset_to, :, :]
                test_dict['Y'] = test_dict['Y'][0:DATASET.trunc_testset_to, :]

        if not validation and not test:
            return data_dict
        elif not test:
            return data_dict, validation_dict
        else: 
            return data_dict, validation_dict, test_dict
    else:
        print( "Unknown dataset")
        exit()