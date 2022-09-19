import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
print(tf.__version__)
from keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import VGG19,Xception
from tensorflow.keras.layers import Input,Flatten,Dense,BatchNormalization,Activation,Dropout,GlobalAveragePooling2D,MaxPooling2D,RandomFlip,RandomZoom,RandomRotation
from keras.datasets import cifar10


from tensorflow.keras.losses import MSE

def generate_image_adversary(model, image, label, eps=1e-6):
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = MSE(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adversary = (image + eps*signed_grad).numpy()
    return signed_grad


(x_train, y_train), (x_val, y_val) = cifar10.load_data()

y_train=to_categorical(y_train)
y_val=to_categorical(y_val)

print((x_train.shape, y_train.shape))
print((x_val.shape, y_val.shape))


base_model = Xception(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=y_train.shape[1])

base_model.summary()



inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (224,224)))(inputs)
x = tf.keras.applications.xception.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation=('softmax'))(x)
model = tf.keras.Model(inputs, outputs)


# In[12]:


# Check the architecture of the final model

model.summary()


# In[ ]:





# In[13]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


epochs = 20
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, verbose=1)


# ## Cross Validation
# 
# Source: https://www.kaggle.com/ryanholbrook/the-convolutional-classifier

# In[ ]:


def plot_history(history):
    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    
    return
    
plot_history(history)


# # Fine Tuning

# In[ ]:


# unfreeze the layers of the pre-trained model

base_model.trainable = True


# In[ ]:


# Use a small learning rate

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


epochs = 10
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, verbose=1)


# ## Cross Validation

# In[ ]:


plot_history(history)


# # Confusion Matrix

# In[ ]:


class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

predictions=model.predict(x_val)

y_pred_classes = np.argmax(predictions, axis=1)
y_true = np.argmax(y_val, axis=1)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12, 9))
c = sns.heatmap(confusion_mtx, annot=True, fmt='g')
c.set(xticklabels=class_names, yticklabels=class_names)

