
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

'''
python train.py --train=yes
'''

## Optimize training hyperparameters
1. For this section, you'll need to install first the optional dependencies:
'''
pip install hyperopt, pymongo, networkx
'''

2. Lunch the hyperparamets search:
'''
python optimize_hyperparams.py --max_evals=20
'''

3. You should then retrain your model with the best parameters

