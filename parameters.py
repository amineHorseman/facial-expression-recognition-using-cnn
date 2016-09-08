
import os

class Dataset:
    name = 'Fer2013'
    train_folder = '../fer2013/fer2013_5000x5/train_set'
    validation_folder = '../fer2013/fer2013_5000x5/validation_set'
    test_folder = '../fer2013/fer2013_5000x5/test_set'
    trunc_trainset_to = 500
    trunc_validationset_to = 100
    trunc_testset_to = -1

class Network:
    input_size = 48
    output_size = 5
    activation = 'relu'
    loss = 'categorical_crossentropy'
    use_landmarks = True

class Hyperparams:
    keep_prob = 0.8
    learning_rate = 0.001
    learning_rate_decay = 0.96
    decay_step = 50
    optimizer = 'momentum'  # {'momentum', 'adam'}
    optimizer_param = 0.9   # momentum value for Momentum optimizer, or beta1 value for Adam

class Training:
    batch_size = 128
    epochs = 1
    snapshot_step = 100
    vizualize = True
    logs_dir = "logs"
    checkpoint_dir = "checkpoints/"
    best_checkpoint_path = "checkpoints/best/"
    max_checkpoints = 1
    checkpoint_frequency = 1.0 # in hours
    save_model = False
    save_model_path = "checkpoints/saved_model.bin"

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

DATASET = Dataset()
NETWORK = Network()
TRAINING = Training()
HYPERPARAMS = Hyperparams()

make_dir(TRAINING.logs_dir)
make_dir(TRAINING.checkpoint_dir)