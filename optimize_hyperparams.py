"""
@AmineHorseman
Sep, 7th, 2016
"""
import time
import argparse
import pprint
import numpy as np 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from train import train
from parameters import HYPERPARAMS, OPTIMIZER

# define the search space
fspace = {
    'learning_rate': hp.uniform('learning_rate', OPTIMIZER.learning_rate['min'], OPTIMIZER.learning_rate['max']),
    'learning_rate_decay': hp.uniform('learning_rate_decay', OPTIMIZER.learning_rate_decay['min'], OPTIMIZER.learning_rate_decay['max']),
    'optimizer': hp.choice('optimizer', OPTIMIZER.optimizer),
    'optimizer_param': hp.uniform('optimizer_param', OPTIMIZER.optimizer_param['min'], OPTIMIZER.optimizer_param['max']),
    'keep_prob': hp.uniform('keep_prob', OPTIMIZER.keep_prob['min'], OPTIMIZER.keep_prob['max'])
}

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--max_evals", required=True, help="Maximum number of evaluations during hyperparameters search")
args = parser.parse_args()
max_evals = int(args.max_evals)
current_eval = 1
train_history = []

# defint the fucntion to minimize (will train the model using the specified hyperparameters)
def function_to_minimize(hyperparams, optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param, 
        learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob, 
        learning_rate_decay=HYPERPARAMS.learning_rate_decay):
    if 'learning_rate' in hyperparams: 
        learning_rate = hyperparams['learning_rate']
    if 'learning_rate_decay' in hyperparams: 
        learning_rate_decay = hyperparams['learning_rate_decay']
    if 'keep_prob' in hyperparams: 
        keep_prob = hyperparams['keep_prob']
    if 'optimizer' in hyperparams:
        optimizer = hyperparams['optimizer']
    if 'optimizer_param' in hyperparams:
        optimizer_param = hyperparams['optimizer_param']
    global current_eval 
    global max_evals
    print( "#################################")
    print( "       Evaluation {} of {}".format(current_eval, max_evals))
    print( "#################################")
    start_time = time.time()
    try:
        accuracy = train(learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, 
                     optimizer=optimizer, optimizer_param=optimizer_param, keep_prob=keep_prob)
        training_time = int(round(time.time() - start_time))
        current_eval += 1
        train_history.append({'accuracy':accuracy, 'learning_rate':learning_rate, 'learning_rate_decay':learning_rate_decay, 
                                  'optimizer':optimizer, 'optimizer_param':optimizer_param, 'keep_prob':keep_prob, 'time':training_time})
    except Exception as e:
        # exception occured during training, saving history and stopping the operation
        print( "#################################")
        print( "Exception during training: {}".format(str(e)))
        print( "Saving train history in train_history.npy")
        np.save("train_history.npy", train_history)
        exit()
    return {'loss': -accuracy, 'time': training_time, 'status': STATUS_OK}

# lunch the hyperparameters search
trials = Trials()
best_trial = fmin(fn=function_to_minimize, space=fspace, algo=tpe.suggest, max_evals=max_evals, trials=trials)

# get some additional information and print the best parameters
for trial in trials.trials:
    if trial['misc']['vals']['keep_prob'][0] == best_trial['keep_prob'] and \
            trial['misc']['vals']['learning_rate'][0] == best_trial['learning_rate'] and \
            trial['misc']['vals']['learning_rate_decay'][0] == best_trial['learning_rate_decay']:
        best_trial['accuracy'] = -trial['result']['loss'] * 100
        best_trial['time'] = trial['result']['time']
print( "#################################")
print( "      Best parameters found")
print( "#################################")
pprint.pprint(best_trial)
print( "#################################")
