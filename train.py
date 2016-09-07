
from tflearn import DNN

from data_loader import load_data 
from parameters import DATASET, TRAINING
from model import build_model

print "loading dataset " + DATASET.name + "..."
data, validation = load_data(validation=True)

print "building model..."
network = build_model()
model = DNN(network, tensorboard_dir=TRAINING.logs_dir, 
        tensorboard_verbose=0, checkpoint_path=TRAINING.checkpoint_dir,
        max_checkpoints=TRAINING.max_checkpoints)

#tflearn.config.init_graph(seed=None, log_device=False, num_cores=6)

print "start training..."
"""model.fit({'input1': data['X']}, {'output': data['Y']}, n_epoch=TRAINING.epochs, 
        validation_set=({'input': validation['X']}, {'output': validation['Y']}),
        snapshot_step=TRAINING.snapshot_step, show_metric=TRAINING.vizualize,
        batch_size=TRAINING.batch_size)
"""

print data['Y'].shape
model.fit([data['X'], data['X2']], data['Y'],
        validation_set=([validation['X'], validation['X2']], validation['Y']),
        show_metric=TRAINING.vizualize,
        batch_size=TRAINING.batch_size) 

print "saving model..."
model.save(TRAINING.save_model_path)