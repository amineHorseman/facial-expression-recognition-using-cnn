"""
@AmineHorseman
Sep, 12th, 2016
"""
import tensorflow as tf
from tflearn import DNN
import time
import numpy as np
import argparse
import dlib
import cv2
import os

from parameters import DATASET, TRAINING, HYPERPARAMS, NETWORK
from model import build_model

def load_model():

    model = None
    with tf.Graph().as_default():
        print "loading pretrained model..."
        network = build_model()
        model = DNN(network, tensorboard_dir=TRAINING.logs_dir, 
                    tensorboard_verbose=0, checkpoint_path=TRAINING.checkpoint_dir,
                    max_checkpoints=TRAINING.max_checkpoints)
        if os.path.isfile(TRAINING.save_model_path):
            model.load(TRAINING.save_model_path)
        else:
            print "Error: file '{}' not found".format(TRAINING.save_model_path)
    return model

def get_landmarks(image, rects, predictor):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def predict(model, image, shape_predictor=None):
    # get landmarks
    if NETWORK.use_landmarks:
        face_rects = [dlib.rectangle(left=0, top=0, right=NETWORK.input_size, bottom=NETWORK.input_size)]
        face_landmarks = np.array([get_landmarks(image, face_rects, shape_predictor)])
        flatten_image = image.reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        predicted_label = model.predict([flatten_image, face_landmarks])
    else:
        predicted_labels = model.predict(flatten_image)
    return get_emotion(predicted_label[0])

def get_emotion(label):
    print "- Angry: {0:.1f}%\n- Happy: {1:.1f}%\n- Sad: {2:.1f}%\n- Surprise: {3:.1f}%\n- Neutral: {4:.1f}%".format(
        label[0]*100, label[1]*100, label[2]*100, label[3]*100, label[4]*100
    )

# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="Image file to predict")
args = parser.parse_args()
if args.image:
    if os.path.isfile(args.image):
        model = load_model()
        image = cv2.imread(args.image, 0)
        shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
        start_time = time.time()
        emotion = predict(model, image, shape_predictor)
        total_time = time.time() - start_time
        print "time: {0:.1f} sec".format(total_time)
    else:
        print "Error: file '{}' not found".format(args.image)