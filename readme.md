
# Facial expression (emotion) recognition for AI Samurai v2



# Dependencies

- Tensorflow
- Tflearn
- Numpy
- Argparse
- [optional] Hyperopt + pymongo + networkx

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
1. For this section, you'll need to install first the optional dependencies:
```
pip install hyperopt, pymongo, networkx
```

2. Lunch the hyperparamets search:
```
python optimize_hyperparams.py --max_evals=20
```

3. You should then retrain your model with the best parameters

N.B: the accuracies displayed is for validation_set (not test_set)

### Evaluate model (test phase)

1. Modify 'parameters.py' set:

Set "save_model_path" parameter to the path of your pretrained file

2. Launch evaluation on test_set:

```
python train.py --evaluate=yes
```
