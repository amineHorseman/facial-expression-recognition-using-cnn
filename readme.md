
# Facial expression (emotion) recognition for AI Samurai v2

TODO: Add description here

# Dependencies

- Tensorflow
- Tflearn
- Numpy
- Argparse
- [optional] Hyperopt + pymongo + networkx
- [optional] dlib, imutils, opencv 3

# HOW TO USE?

## Train the model
1. Choose your parameters in 'parameters.py'

2. Launch training:

```
python train.py --train=yes
```

3. Train and evaluate:

```
python train.py --train=yes --evaluate=yes
```

N.B: make sure the parameter "save_model" (in parameters.py) is set to True if you want to train and evaluate

## Optimize training hyperparameters
1. For this section, you'll need to install first these optional dependencies:
```
pip install hyperopt, pymongo, networkx
```

2. Lunch the hyperparamets search:
```
python optimize_hyperparams.py --max_evals=20
```

3. You should then retrain your model with the best parameters

N.B: the accuracies displayed is for validation_set (not test_set)

### Evaluate model (calculating test accuracy)

1. Modify 'parameters.py':
 
Set "save_model_path" parameter to the path of your pretrained file

2. Launch evaluation on test_set:

```
python train.py --evaluate=yes
```

### Recognizing facial expressions from an image file

1. For this section you will need to install `dlib` and `opencv 3` dependencies

2. Modify 'parameters.py':

Set "save_model_path" parameter to the path of your pretrained file

3. Predict emotions from a file

```
python predict.py --image path/to/image.jpg
```

### Recognizing facial expressions in real time from video

1. For this section you will need to install `dlib`, `imutils` and `opencv 3` dependencies

2. Modify 'parameters.py':

Set "save_model_path" parameter to the path of your pretrained file

3. Predict emotions from a file

```
python predict-from-video.py
```
