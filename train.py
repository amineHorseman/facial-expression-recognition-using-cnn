
from tflearn import DNN
import time
import argparse

from data_loader import load_data 
from parameters import DATASET, TRAINING, HYPERPARAMS
from model import build_model

def train(optimizer=HYPERPARAMS.optimizer, learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob):
        print "loading dataset " + DATASET.name + "..."
        data, validation = load_data(validation=True)

        print "building model..."
        network = build_model(optimizer, learning_rate, keep_prob)
        model = DNN(network, tensorboard_dir=TRAINING.logs_dir, 
                tensorboard_verbose=0, checkpoint_path=TRAINING.checkpoint_dir,
                max_checkpoints=TRAINING.max_checkpoints)

        #tflearn.config.init_graph(seed=None, log_device=False, num_cores=6)

        print "start training..."
        start_time = time.time()
        """model.fit({'input1': data['X']}, {'output': data['Y']}, n_epoch=TRAINING.epochs, 
                validation_set=({'input': validation['X']}, {'output': validation['Y']}),
                snapshot_step=TRAINING.snapshot_step, show_metric=TRAINING.vizualize,
                batch_size=TRAINING.batch_size)
        """
        model.fit([data['X'], data['X2']], data['Y'],
                validation_set=([validation['X'], validation['X2']], validation['Y']),
                show_metric=TRAINING.vizualize,
                batch_size=TRAINING.batch_size)
        training_time = time.time() - start_time
        print "training time = " + training_time

        if TRAINING.save_model:
                print "saving model..."
                model.save(TRAINING.save_model_path)

        return accuracy, training_time

# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no", help="if 'yes', launch training from command line")
args = parser.parse_args()
if args.train=="yes" or args.train=="Yes" or args.train=="YES":
        train()