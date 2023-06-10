import urllib
from urllib import request, error
import sys
import subprocess

# Download the Face Landmarks model
data_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
filename = data_url.split('/')[-1]


# DownLoad model
def _progress(block_num, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                     float(block_num * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()


# Download model when network conditions is not good
def re_down(url, filename, _progress):
    try:
        urllib.request.urlretrieve(url,filename,_progress)
    except urllib.error.ContentTooShortError:
        print ('Network conditions is not good. Reloading...')
        re_down(url,filename,_progress)


if __name__ == '__main__':
    re_down(data_url, filename, _progress)
    # Download Dataset,put kaggle.json in .kaggle
    subprocess.call(
        'kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge',
        shell=True)


