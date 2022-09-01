import numpy as np
from skimage.transform import rotate
from scipy.ndimage import shift
from sklearn.metrics import confusion_matrix
import random

def random_rotate(array, max_angle):
    if random.random() > 0.5:
        array_out = np.zeros(array.shape)
        
        random_angle = random.randint(-max_angle, max_angle)

        for i in range(array.shape[0]):
            array_out[i] = rotate(array[i], random_angle, preserve_range=True)

        return array_out
    else:
        return array

def random_shift(array, max_shift):
    if random.random() > 0.5:
        array_out = np.zeros(array.shape)
        
        random_x = random.randint(-max_shift, max_shift)
        random_y = random.randint(-max_shift, max_shift)

        for i in range(array.shape[0]):
            array_out[i] = shift(array[i], (random_x, random_y))

        return array_out
    else:
        return array

def random_flip(array):
    if random.random() > 0.5:
        array_out = np.zeros(array.shape)
        
        for i in range(array.shape[0]):
            array_out[i] = np.fliplr(array[i])

        return array_out
    else:
        return array

def accuracy_sensitivity_specificity(y_trues, y_preds):
    cm = confusion_matrix(y_trues, y_preds)
    tn, fp, fn, tp = cm.ravel()
    total = sum(sum(cm))
    #accuracy = (cm[0,0] + cm[1,1]) / total
    #sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    #specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    accuracy = (tp + tn) / total
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, sensitivity, specificity