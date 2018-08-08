
# Facial expression recognition using CNN in Tensorflow

The package uses a convolutional neural network to classify images from files or from video/camera1 stream.

## Motivation

The goal is to get a quick baseline to compare if the CNN architecture performs better when it uses only the raw pixels of images for training, or if it's better to feed some extra information to the CNN (such as face landmarks or HOG features). The results show that the extra information helps the CNN to perform better.

To train the model, we use Fer2013 datset that contains 30,000 images of expressions grouped in seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral.
The problem challenge is that Fer2013 images are not aligned and it's difficult to classify facial expressions from it. The faces are first detected using opencv, then we extract the face landmarks using dlib. We also extracted the HOG features and we input the raw image data with the face landmarks+hog into a 4 layered convolutional neural network.

![Model's architecture](https://AmineHorseman/model_architecture.png)

## Classification Results:

|       Experiments                                        |   5 emotions  |
|----------------------------------------------------------|---------------|
| SVM + HOG features                                       |     34.4%     |
| SVM + Face Landmarks                                     |     46.9%     |
| SVM + HOG + Face landmarks                               |     55.0%     |
| SVM + HOG + Face landmarks + sliding window              |     59.4%     |
| CNN (on raw pixels)                                      |     65.0%     |
| CNN + Face landmarks                                     |     47.3%     |
| CNN + Face landmarks + HOG                               |     66.8%     |
| CNN + Face landmarks + HOG + batch norm                  |     68.7%     |
| CNN + Face landmarks + HOG + sliding window + batch norm |     71.4%     |

For the experiments we used only 5 expressions: Angry, Happy, Sad, Surprise, Neutral.

The comparison with SVM was done using the results from this repository:
[Facial Expressions Recognition using SVM](https://github.com/amineHorseman/facial-expression-recognition-svm)


## HOW TO USE?

### Install dependencies

- Tensorflow
- Tflearn
- Numpy
- Argparse
- [optional] Hyperopt + pymongo + networkx
- [optional] dlib, imutils, opencv 3
- [optional] scipy, pandas, skimage

Better to use anaconda environemnt to easily install the dependencies (especially opencv and dlib)

### Download and prepare the data

1. Download Fer2013 dataset and the Face Landmarks model

    - [Kaggle Fer2013 challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
    - [Dlib Shape Predictor model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

2. Unzip the downloaded files

    And put the files `fer2013.csv` and `shape_predictor_68_face_landmarks.dat` in the root folder of this package.

3. Convert the dataset to extract Face Landmarks and HOG Features
    ```
    python convert_fer2013_to_images_and_landmarks.py
    ```

    You can also use these optional arguments according to your needs:
    `-j`, `--jpg` (yes|no): **save images as .jpg files (default=no)**
    `-l`, `--landmarks` *(yes|no)*: **extract Dlib Face landmarks (default=yes)**
    `-ho`, `--hog` (yes|no): **extract HOG features (default=yes)**
    `-hw`, `--hog_windows` (yes|no): **extract HOG features using a sliding window (default=yes)**
    `-o`, `--onehot` (yes|no): **one hot encoding (default=yes)**
    `-e`, `--expressions` (list of numbers): **choose the faciale expression you want to use: *0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral* (default=0,1,2,3,4,5,6)**

    Examples
    ```
    python convert_fer2013_to_images_and_landmarks.py
    python convert_fer2013_to_images_and_landmarks.py --landmarks=yes --hog=no --how_windows=no --jpg=no --expressions=1,4,6
    ```
    The script will create a folder with the data prepared and saved as numpy arrays.
    Make sure the `--onehot` argument set to `yes` (default value)

### Train the model
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

### Optimize training hyperparameters
1. For this section, you'll need to install first these optional dependencies:
```
pip install hyperopt, pymongo, networkx
```

2. Lunch the hyperparamets search:
```
python optimize_hyperparams.py --max_evals=20
```

3. You should then retrain your model with the best parameters

N.B: the accuracies displayed are for validation_set only (not test_set)

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
A window will appear with a box around the face and the predicted expression.
Press 'q' key to stop.

## TODO
Some ideas for interessted contributors:
- Python 3 compatibility
- Change model's architecture?
- Add other features extraction techniques?
- Add more datasets?
