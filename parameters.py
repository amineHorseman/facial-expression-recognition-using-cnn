"""
@AmineHorseman
Sep, 1st, 2016
"""
import os

class Dataset:
    name = 'Fer2013'
    train_folder = '../fer2013/fer2013_30000x5/train_set'
    validation_folder = '../fer2013/fer2013_30000x5/validation_set'
    test_folder = '../fer2013/fer2013_30000x5/test_set'
    shape_predictor_path='../fer2013/shape_predictor_68_face_landmarks.dat'
    trunc_trainset_to = 500
    trunc_validationset_to = 500
    trunc_testset_to = -1

class Network:
    input_size = 48
    output_size = 5
    activation = 'relu'
    loss = 'categorical_crossentropy'
    use_landmarks = False
    use_hog_and_landmarks = True
    use_batchnorm_after_conv_layers = False
    use_batchnorm_after_fully_connected_layers = False

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
    snapshot_step = 10000
    vizualize = True
    logs_dir = "logs"
    checkpoint_dir = "checkpoints/"
    best_checkpoint_path = "checkpoints/best/"
    max_checkpoints = 1
    checkpoint_frequency = 1.0 # in hours
    save_model = True
    save_model_path = "checkpoints/saved_model.bin"

class VideoPredictor:
    emotions = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]
    print_emotions = False
    send_by_osc_socket = False
    send_by_socket = False
    ip = "127.0.0.1"    # destination address for sockets and OSC sockets 
    port = 9003
    camera_source = 0
    face_detection_classifier = "lbpcascade_frontalface.xml"
    show_confidence = False
    time_to_wait_between_predictions = 0.5
    time_to_wait_to_send_by_socket = 5

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

DATASET = Dataset()
NETWORK = Network()
TRAINING = Training()
HYPERPARAMS = Hyperparams()
VIDEO_PREDICTOR = VideoPredictor()

make_dir(TRAINING.logs_dir)
make_dir(TRAINING.checkpoint_dir)