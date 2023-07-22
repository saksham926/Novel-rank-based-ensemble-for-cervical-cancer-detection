import os,glob
import numpy as np
import os
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import cv2
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
import glob
import pickle
import tensorflow as tf
import argparse
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import datetime
import keras
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,Layer,ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
#from keras_cv.layers import RandomCutout
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from skimage.transform import radon, rescale
from skimage.filters import roberts, sobel, scharr, prewitt
from classification_models.keras import Classifiers
from skimage import feature
import os,glob
import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
import pickle
import argparse
import re
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as rp
from classification_models.keras import Classifiers
from tensorflow.keras.applications.xception import preprocess_input as xp
import sklearn.metrics as metrics
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.utils import get_file
import os,glob
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import keras
from classification_models.keras import Classifiers
import numpy as np
import cv2
import glob
import pickle
#import clahe
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix , classification_report
from matplotlib import pyplot as plt
import tensorflow as tf
import argparse
import re
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from sympy.solvers import solve
from sympy import Symbol
import seaborn as sns
import numpy as np
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,Layer,ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from skimage.transform import radon, rescale
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature
import os,glob
import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
import pickle
import argparse
import re
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from tensorflow.keras.layers import Layer
from PIL import Image
from numpy import asarray
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf   
import keras
from classification_models.keras import Classifiers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from tensorflow.keras.metrics import Recall, Precision
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from tensorflow.keras.metrics import Recall, Precision
from skimage import data, exposure
from tensorflow.keras.layers import Layer
from PIL import Image
from numpy import asarray
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import pandas as pd

import cv2
IM_SIZE=224
#import tensorflow_probability as tfp


def no_data_augmentation(cat1_files,cat2_files,cat3_files,cat4_files,cat5_files):
    aug_cat1=[]
    aug_cat2=[]
    aug_cat3=[]
    aug_cat4=[]
    aug_cat5=[]
    for ele in cat1_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(224,224),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_cat1.append(pic)
    for ele in cat2_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(224,224),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_cat2.append(pic)
    for ele in cat3_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
      
        pic = cv2.resize(x,(224,224),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_cat3.append(pic)
    
    for ele in cat4_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(224,224),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_cat4.append(pic)    
    for ele in cat5_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(224,224),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_cat5.append(pic)   
    for i in range(len(aug_cat1)):
        aug_cat1[i]=aug_cat1[i].reshape((224,224,3))
    
    for i in range(len(aug_cat2)):
        aug_cat2[i]=aug_cat2[i].reshape((224,224,3))
    for i in range(len(aug_cat3)):
        aug_cat3[i]=aug_cat3[i].reshape((224,224,3))
    for i in range(len(aug_cat4)):
        aug_cat4[i]=aug_cat4[i].reshape((224,224,3))    
    for i in range(len(aug_cat5)):
        aug_cat5[i]=aug_cat5[i].reshape((224,224,3)) 
    
    print("Category 1 files without augmentation:",len(aug_cat1))
    print("Category 2 files without augmentation:",len(aug_cat2))
    print("Category 3 files without augmentation:",len(aug_cat3))
    print("Category 4 files without augmentation:",len(aug_cat4))
    print("Category 5 files without augmentation:",len(aug_cat5))
    return aug_cat1,aug_cat2,aug_cat3, aug_cat4, aug_cat5


def box(lamda):
    IM_SIZE=224
    r_x=int(np.random.uniform(0, IM_SIZE))
    
    
    r_y=int(np.random.uniform(0, IM_SIZE))
    
    r_w=IM_SIZE*np.sqrt(1 - lamda)
    r_h=IM_SIZE*np.sqrt(1 - lamda)

    r_x=np.clip(r_x - r_w // 2, 0, IM_SIZE)
    r_y=np.clip(r_y-r_h//2, 0, IM_SIZE)

    x_b_r=np.clip(r_x+r_w//2,0, IM_SIZE)
    y_b_r=np.clip(r_y+r_h//2,0, IM_SIZE)

    r_w=y_b_r-r_y
    if(r_w==0):
        r_w=1
    r_h=y_b_r-r_y
    if(r_h==0):
        r_h=1
     
    return int(r_y),int(r_x),int(r_h),int(r_w)

  
def cutmix(image1,label1,images,labels):
    np.random.seed(None)
    index = np.random.permutation(len(images))
    lamda=stats.beta(0.4, 0.4).rvs()
    r_y,r_x,r_h,r_w=box(lamda)
    image2 = images[index[0]]
    label2 = labels[index[0]]
    crop2=tf.image.crop_to_bounding_box(image2,r_y,r_x,r_h,r_w)
    pad2=tf.image.pad_to_bounding_box(crop2,r_y,r_x,IM_SIZE,IM_SIZE)
    crop1=tf.image.crop_to_bounding_box(image1,r_y,r_x,r_h,r_w)
    pad1=tf.image.pad_to_bounding_box(crop1,r_y,r_x,IM_SIZE,IM_SIZE)
    image=image1-pad1+pad2
    lamda=1-(r_h*r_w)/(IM_SIZE*IM_SIZE)
    label=lamda*label1+(1-lamda)*label2
    return image,label

def on_hot_encode_labels(lables):
    aug_list=[]
    for i in range(len(lables)):
        if lables[i]==0:
            aug_list.append([0,1,0,0,0])
        elif lables[i]==1:
            aug_list.append([1,0,0,0,0])
        elif lables[i]==2:
            aug_list.append([0,0,1,0,0])
        elif lables[i]==3:
            aug_list.append([0,0,0,1,0])
        elif lables[i]==4:
            aug_list.append([0,0,0,0,1])
    return aug_list
import cv2
def mixup(image1,label1,images,labels):
    index = np.random.permutation(len(images))
    image2 = images[index[0]]
    
    label2 = labels[index[0]]
    lamda=np.random.beta(0.4, 0.4)
    
    label_1=label1
    label_2=label2
    image=lamda*image1+(1-lamda)*image2
    label=lamda*label_1+(1-lamda)*label_2
    
    return image,label

def cutout(images,labels, pad_size=16):
    cut_image=[]
    cut_labels=[]
    for index in tqdm(range(len(images))):
        img=images[index]
        h, w, c = img.shape
        mask = np.ones((h + pad_size*2, w + pad_size*2, c))
        y = np.random.randint(pad_size, h + pad_size)
        x = np.random.randint(pad_size, w + pad_size)
        y1 = np.clip(y - pad_size, 0, h + pad_size*2)
        y2 = np.clip(y + pad_size, 0, h + pad_size*2)
        x1 = np.clip(x - pad_size, 0, w + pad_size*2)
        x2 = np.clip(x + pad_size, 0, w + pad_size*2)
        mask[y1:y2, x1:x2, :] = 0
        img_cutout = img * mask[pad_size:pad_size+h, pad_size:pad_size+w, :]
        cut_image.append(img_cutout)
        cut_labels.append(labels[index])
    return cut_image,cut_labels

def random_augment(image1,label1,images,labels):
    datagen=ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    x=image1.reshape((1,)+image1.shape)
    my_images=[]
    my_labels=[]
    counter=0
    for i in datagen.flow(x):
        if counter==6:
            break
        my_image=i
        my_image=my_image.reshape((224,224,3))
        my_images.append(my_image)
        my_labels.append(label1)
        counter+=1
        
        
    
    return my_images,my_labels
        
    
def advance_data_aug(images_list,images_labels,full_data,full_label,param=2):
    images_list=np.array(images_list)
    images_labels=np.array(images_labels)
    
    # create the original array
    arr = full_label

    # define the value of the element to delete
    value_to_delete = images_labels[0]
    index=[]
    for i in range(len(arr)):
        if np.array_equal(arr[i],value_to_delete):
            index.append(i)
    my_full_data=[]
    my_full_label=[]
    for i in range(len(full_label)):
        if i in index:
            continue
        else:
            my_full_data.append(full_data[i])
            my_full_label.append(full_label[i])
    full_data=my_full_data.copy()
    full_label=my_full_label.copy()
    full_data=np.array(full_data)
    full_label=np.array(full_label)
    aug_list=[]
    aug_labels=[]
    print("adding original images")
    for i in range(len(images_list)):
        aug_labels.append(images_labels[i])
        aug_list.append(images_list[i])
    print(np.array(aug_list).shape,np.array(aug_labels).shape)
    print("cutmix")
    for i in range(2):
        for j in tqdm(range(len(images_list))):
            new_image,new_label=cutmix(images_list[j],images_labels[j],full_data,full_label)
            aug_labels.append(new_label)
            aug_list.append(new_image)
    print(np.array(aug_list).shape,np.array(aug_labels).shape)
    print("mixup")
    for i in range(2):
        for j in tqdm(range(len(images_list))):
            new_image,new_label=mixup(images_list[j],images_labels[j],full_data,full_label)
            aug_labels.append(new_label)
            aug_list.append(new_image)
    print(np.array(aug_list).shape,np.array(aug_labels).shape)
    print("random augmentation")
    for j in tqdm(range(len(images_list))):
        new_image,new_label=random_augment(images_list[j],images_labels[j],full_data,full_label)
        for index in range(len(new_image)):
            aug_labels.append(new_label[index])
            aug_list.append(new_image[index])
    print(np.array(aug_list).shape,np.array(aug_labels).shape)
    print("cutout")
    aug_list=np.array(aug_list)
    aug_labels=np.array(aug_labels)
    for i in range(2):
        im,la=cutout(images_list,images_labels) 
        aug_list = np.concatenate([aug_list, im])
        aug_labels=np.concatenate([aug_labels,la])
        
    print(np.array(aug_list).shape,np.array(aug_labels).shape)
    return aug_list,aug_labels
        
    
    
    
def making_full_data_train(aug_cat1,aug_cat2,aug_cat3,aug_cat4,aug_cat5):
    aug_cat1=shuffle(aug_cat1, random_state=0)
    aug_cat2=shuffle(aug_cat2,random_state=0)
    aug_cat3=shuffle(aug_cat3,random_state=0)
    aug_cat4=shuffle(aug_cat4,random_state=0)
    aug_cat5=shuffle(aug_cat5,random_state=0)
    
    aug_cat1_labels=[]
    for i in range(len(aug_cat1)):
        aug_cat1_labels.append(0)
    print(np.shape(aug_cat1),np.shape(aug_cat1_labels))
    aug_cat2_labels=[]
    for i in range(len(aug_cat2)):
        aug_cat2_labels.append(1)
    print(np.shape(aug_cat2),np.shape(aug_cat2_labels))
    aug_cat3_labels=[]
    for i in range(len(aug_cat3)):
        aug_cat3_labels.append(2)
    print(np.shape(aug_cat3),np.shape(aug_cat3_labels))  
    aug_cat4_labels=[]
    for i in range(len(aug_cat4)):
        aug_cat4_labels.append(3)
    print(np.shape(aug_cat4),np.shape(aug_cat4_labels))  
    aug_cat5_labels=[]
    for i in range(len(aug_cat5)):
        aug_cat5_labels.append(4)
    print(np.shape(aug_cat5),np.shape(aug_cat5_labels)) 
    aug_cat1_labels=on_hot_encode_labels(aug_cat1_labels)
    aug_cat2_labels=on_hot_encode_labels(aug_cat2_labels)
    aug_cat3_labels=on_hot_encode_labels(aug_cat3_labels)
    aug_cat4_labels=on_hot_encode_labels(aug_cat4_labels)
    aug_cat5_labels=on_hot_encode_labels(aug_cat5_labels)
    full_data=[]
    full_label=[]
    for i in range(len(aug_cat1)):
        full_data.append(aug_cat1[i])
        full_label.append(aug_cat1_labels[i])
    for i in range(len(aug_cat2)):
        full_data.append(aug_cat2[i])
        full_label.append(aug_cat2_labels[i])
    for i in range(len(aug_cat3)):
        full_data.append(aug_cat3[i])
        full_label.append(aug_cat3_labels[i])
    for i in range(len(aug_cat4)):
        full_data.append(aug_cat4[i])
        full_label.append(aug_cat4_labels[i])
    for i in range(len(aug_cat5)):
        full_data.append(aug_cat5[i])
        full_label.append(aug_cat5_labels[i])
    aug_cat1,aug_cat1_labels=advance_data_aug(aug_cat1,aug_cat1_labels,full_data,full_label)
    aug_cat2,aug_cat2_labels=advance_data_aug(aug_cat2,aug_cat2_labels,full_data,full_label)
    aug_cat3,aug_cat3_labels=advance_data_aug(aug_cat3,aug_cat3_labels,full_data,full_label)
    aug_cat4,aug_cat4_labels=advance_data_aug(aug_cat4,aug_cat4_labels,full_data,full_label)
    aug_cat5,aug_cat5_labels=advance_data_aug(aug_cat5,aug_cat5_labels,full_data,full_label)

    full_data=[]
    full_label=[]
    for i in range(len(aug_cat1)):
        full_data.append(aug_cat1[i])
        full_label.append(aug_cat1_labels[i])
    for i in range(len(aug_cat2)):
        full_data.append(aug_cat2[i])
        full_label.append(aug_cat2_labels[i])
    for i in range(len(aug_cat3)):
        full_data.append(aug_cat3[i])
        full_label.append(aug_cat3_labels[i])
    for i in range(len(aug_cat4)):
        full_data.append(aug_cat4[i])
        full_label.append(aug_cat4_labels[i])
    for i in range(len(aug_cat5)):
        full_data.append(aug_cat5[i])
        full_label.append(aug_cat5_labels[i])
        
    full_data=np.array(full_data)
    full_label=np.array(full_label)
    
    full_data=shuffle(full_data,random_state=0)
    full_label=shuffle(full_label,random_state=0)
    
    return full_data,full_label
"""Inception 2D_CNN Models in Tensorflow-Keras.
References -
Inception_v1 (GoogLeNet): https://arxiv.org/abs/1409.4842 [Going Deeper with Convolutions]
Inception_v2: http://arxiv.org/abs/1512.00567 [Rethinking the Inception Architecture for Computer Vision]
Inception_v3: http://arxiv.org/abs/1512.00567 [Rethinking the Inception Architecture for Computer Vision]
Inception_v4: https://arxiv.org/abs/1602.07261 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning]
"""




def Conv_2D_Block(x, model_width, kernel, strides=(1, 1), padding="same"):
    # 2D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)
    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs         : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)
    return out


def SE_Block(inputs, num_filters, ratio):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    excitation = tf.keras.layers.Dense(units=num_filters/ratio)(squeeze)
    excitation = tf.keras.layers.Activation('relu')(excitation)
    excitation = tf.keras.layers.Dense(units=num_filters)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([1, 1, num_filters])(excitation)

    scale = inputs * excitation

    return scale


def Inceptionv1_Module(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1), padding='valid')

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1), padding='valid')
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 3))

    branch5x5 = Conv_2D_Block(inputs, filterB3_1, (1, 1), padding='valid')
    branch5x5 = Conv_2D_Block(branch5x5, filterB3_2, (5, 5))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))
    out = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_Block_'+str(i))

    return out


def Inceptionv2_Module(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 3))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 3))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_'+str(i))

    return out


def Inception_Module_A(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch5x5 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch5x5 = Conv_2D_Block(branch5x5, filterB2_2, (5, 5))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 3))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_A'+str(i))

    return out


def Inception_Module_B(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch7x7 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (1, 7))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (7, 1))

    branch7x7dbl = Conv_2D_Block(inputs, filterB3_1, 1)
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_2, (1, 7))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_2, (7, 1))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_3, (1, 7))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_3, (7, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='Inception_Block_B'+str(i))

    return out


def Inception_Module_C(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3_2 = Conv_2D_Block(branch3x3, filterB2_2, (1, 3))
    branch3x3_3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 1))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (1, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 1))
    branch3x3dbl_2 = Conv_2D_Block(branch3x3dbl, filterB3_3, (1, 3))
    branch3x3dbl_3 = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch3x3_2, branch3x3_3, branch3x3dbl_2, branch3x3dbl_3, branch_pool], axis=-1, name='Inception_Block_C'+str(i))

    return out


def Reduction_Block_A(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block A (i)
    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2))

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out


def Reduction_Block_B(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block B (i)
    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2))

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (1, 7))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (7, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out


class SEInception:
    def __init__(self, length, width, num_channel, num_filters, ratio=4, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False, auxilliary_outputs=False):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Model
        # kernel_size: Kernel or Filter Size of the Input Convolutional Layer
        # num_channel: Number of Channels of the Input Predictor Signals
        # problem_type: Regression or Classification
        # output_nums: Number of Output Classes in Classification mode and output features in Regression mode
        # pooling: Choose either 'max' for MaxPooling or 'avg' for Averagepooling
        # dropout_rate: If turned on, some layers will be dropped out randomly based on the selected proportion
        # auxilliary_outputs: Two extra Auxullary outputs for the Inception models, acting like Deep Supervision
        self.length = length
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.ratio = ratio
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.auxilliary_outputs = auxilliary_outputs

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def SEInception_v1(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, self.num_filters, 7, strides=2)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv_2D_Block(x, self.num_filters, 1, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 3, 3)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Inceptionv1_Module(x, 64, 96, 128, 16, 32, 32, 1)  # Inception Block 1
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 128, 128, 192, 32, 96, 64, 2)  # Inception Block 2
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Inceptionv1_Module(x, 192, 96, 208, 16, 48, 64, 3)  # Inception Block 3
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 160, 112, 224, 24, 64, 64, 4)  # Inception Block 4
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 128, 128, 256, 24, 64, 64, 5)  # Inception Block 5
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 112, 144, 288, 32, 64, 64, 6)  # Inception Block 6
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 256, 160, 320, 32, 128, 128, 7)  # Inception Block 7
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 64, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Inceptionv1_Module(x, 256, 160, 320, 32, 128, 128, 8)  # Inception Block 8
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 384, 192, 384, 48, 128, 128, 9)  # Inception Block 9
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v1')

        return model

    def SEInception_v2(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem: 56 x 64
        x = tf.keras.layers.SeparableConv2D(self.num_filters, kernel_size=7, strides=(2, 2), depth_multiplier=1, padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv_2D_Block(x, self.num_filters * 2, 1, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 6, 3, padding='valid')
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Inceptionv2_Module(x, 64, 64, 64, 64, 96, 96, 32, 1)  # Inception Block 1: 28 x 192
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv2_Module(x, 64, 64, 96, 64, 96, 96, 64, 2)  # Inception Block 2: 28 x 256
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 128, 160, 64, 96, 96, 1)  # Reduction Block 1: 28 x 320

        x = Inceptionv2_Module(x, 224, 64, 96, 96, 128, 128, 128, 3)  # Inception Block 3: 14 x 576
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv2_Module(x, 192, 96, 128, 96, 128, 128, 128, 4)  # Inception Block 4: 14 x 576
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv2_Module(x, 160, 128, 160, 128, 160, 160, 96, 5)  # Inception Block 5: 14 x 576
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv2_Module(x, 96, 128, 192, 160, 192, 192, 96, 6)  # Inception Block 6: 14 x 576
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 192, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 128, 192, 192, 256, 256, 2)  # Reduction Block 2: 14 x 576

        x = Inceptionv2_Module(x, 352, 192, 320, 160, 224, 224, 128, 7)  # Inception Block 7: 7 x 1024
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv2_Module(x, 352, 192, 320, 192, 224, 224, 128, 8)  # Inception Block 8: 7 x 1024
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v2')

        return model

    def SEInception_v3(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, self.num_filters, 3, strides=2, padding='valid')
        x = Conv_2D_Block(x, self.num_filters, 3, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 2, 3)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Conv_2D_Block(x, self.num_filters * 2.5, 1, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 6, 3, padding='valid')
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # 3x Inception-A Blocks
        x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 32, 1)  # Inception-A Block 1: 35 x 256
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 64, 2)  # Inception-A Block 2: 35 x 256
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 64, 3)  # Inception-A Block 3: 35 x 256
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 64, 384, 64, 96, 96, 1)  # Reduction Block 1: 17 x 768

        # 4x Inception-B Blocks
        x = Inception_Module_B(x, 192, 128, 192, 128, 128, 192, 192, 1)  # Inception-B Block 1: 17 x 768
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_B(x, 192, 160, 192, 160, 160, 192, 192, 2)  # Inception-B Block 2: 17 x 768
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_B(x, 192, 160, 192, 160, 160, 192, 192, 3)  # Inception-B Block 3: 17 x 768
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_B(x, 192, 192, 192, 192, 192, 192, 192, 4)  # Inception-B Block 4: 17 x 768
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 192, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 192, 320, 192, 192, 192, 2)  # Reduction Block 2: 8 x 1280

        # 2x Inception-C Blocks: 8 x 2048
        x = Inception_Module_C(x, 320, 384, 384, 448, 384, 384, 192, 1)  # Inception-C Block 1: 8 x 2048
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_C(x, 320, 384, 384, 448, 384, 384, 192, 2)  # Inception-C Block 2: 8 x 2048
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v3')

        return model

    def SEInception_v4(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, 32, 3, strides=2, padding='valid')
        x = Conv_2D_Block(x, 32, 3, padding='valid')
        x = Conv_2D_Block(x, 64, 3)

        branch1 = Conv_2D_Block(x, 96, 3, strides=2, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = Conv_2D_Block(x, 64, 1)
        branch1 = Conv_2D_Block(branch1, 96, 3, padding='valid')
        branch2 = Conv_2D_Block(x, 64, 1)
        branch2 = Conv_2D_Block(branch2, 64, 7)
        branch2 = Conv_2D_Block(branch2, 96, 3, padding='valid')
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = Conv_2D_Block(x, 192, 3, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        # 4x Inception-A Blocks - 35 x 256
        for i in range(4):
            x = Inception_Module_A(x, 96, 64, 96, 64, 96, 96, 96, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 96, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 64, 384, 192, 224, 256, 1)  # Reduction Block 1: 17 x 768

        # 7x Inception-B Blocks - 17 x 768
        for i in range(7):
            x = Inception_Module_B(x, 384, 192, 256, 192, 224, 256, 128, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 192, 192, 256, 320, 320, 2)  # Reduction Block 2: 8 x 1280

        # 3x Inception-C Blocks: 8 x 2048
        for i in range(3):
            x = Inception_Module_C(x, 256, 384, 512, 384, 512, 512, 256, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v4')
        if self.auxilliary_outputs:
            model = tf.keras.layers.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v4')

        return model


accuracy_gamma=[]
precision_gamma=[]
recall_gamma=[]
fscore_gamma=[]
accuracy_rank=[]
precision_rank=[]
recall_rank=[]
fscore_rank=[]
accuracy_sugeno=[]
precision_sugeno=[]
recall_sugeno=[]
fscore_sugeno=[]
accuracy_weighted=[]
precision_weighted=[]
recall_weighted=[]
fscore_weighted=[]
    
def making_training_and_testing_data_train(full_data,full_label):
    return full_data,full_label
    
def my_plots(folder_path,history,my_model):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    my_path="training and validation accuracy curve of "+my_model+".png"
    plt.savefig(folder_path+my_path)
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim([0, 1])

    #plt.ylim([-3, 3])
    plt.yticks(np.arange(0, 1.1, 0.25))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    my_path="training and validation loss curve of "+my_model+".png"
    plt.savefig(folder_path+my_path)
    plt.show()
    

def making_training_and_testing_data(full_data,full_label):
    train_label=[]
    for i in range(len(full_label)):
        if full_label[i]==0:
            train_label.append([0,1,0,0,0])
        elif full_label[i]==1:
            train_label.append([1,0,0,0,0])
        elif full_label[i]==2:
            train_label.append([0,0,1,0,0])
        elif full_label[i]==3:
            train_label.append([0,0,0,1,0])
        elif full_label[i]==4:
            train_label.append([0,0,0,0,1])
    full_label=np.array(train_label)
    return full_data,full_label
def making_full_data(aug_cat1,aug_cat2,aug_cat3,aug_cat4,aug_cat5):
    aug_cat1=shuffle(aug_cat1, random_state=0)
    aug_cat2=shuffle(aug_cat2,random_state=0)
    aug_cat3=shuffle(aug_cat3,random_state=0)
    aug_cat4=shuffle(aug_cat4,random_state=0)
    aug_cat5=shuffle(aug_cat5,random_state=0)
    
    aug_cat1_labels=[]
    for i in range(len(aug_cat1)):
        aug_cat1_labels.append(0)
    print(np.shape(aug_cat1),np.shape(aug_cat1_labels))
    aug_cat2_labels=[]
    for i in range(len(aug_cat2)):
        aug_cat2_labels.append(1)
    print(np.shape(aug_cat2),np.shape(aug_cat2_labels))
    aug_cat3_labels=[]
    for i in range(len(aug_cat3)):
        aug_cat3_labels.append(2)
    print(np.shape(aug_cat3),np.shape(aug_cat3_labels))  
    aug_cat4_labels=[]
    for i in range(len(aug_cat4)):
        aug_cat4_labels.append(3)
    print(np.shape(aug_cat4),np.shape(aug_cat4_labels))  
    aug_cat5_labels=[]
    for i in range(len(aug_cat5)):
        aug_cat5_labels.append(4)
    print(np.shape(aug_cat5),np.shape(aug_cat5_labels)) 

    full_data=[]
    full_label=[]
    for i in range(len(aug_cat1)):
        full_data.append(aug_cat1[i])
        full_label.append(aug_cat1_labels[i])
    for i in range(len(aug_cat2)):
        full_data.append(aug_cat2[i])
        full_label.append(aug_cat2_labels[i])
    for i in range(len(aug_cat3)):
        full_data.append(aug_cat3[i])
        full_label.append(aug_cat3_labels[i])
    for i in range(len(aug_cat4)):
        full_data.append(aug_cat4[i])
        full_label.append(aug_cat4_labels[i])
    for i in range(len(aug_cat5)):
        full_data.append(aug_cat5[i])
        full_label.append(aug_cat5_labels[i])
        
    full_data=np.array(full_data)
    full_label=np.array(full_label)
    
    full_data=shuffle(full_data,random_state=0)
    full_label=shuffle(full_label,random_state=0)
    
    return full_data,full_label
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as rp
from classification_models.keras import Classifiers
from tensorflow.keras.applications.xception import preprocess_input as xp
import sklearn.metrics as metrics
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
 
if __name__ == '__main__':  #straight away go to this
    cat1_dir = "im_Dyskeratotic/CROPPED/" 
    dir1 = os.path.join(cat1_dir,"*.bmp")
    dir2 = os.path.join(cat1_dir,"*.jpeg")
    dir = os.path.join(cat1_dir,"*.jpg")
    cat1_files = glob.glob(dir)
    cat1_1 = glob.glob(dir1)
    cat1_2 = glob.glob(dir2)
    cat1_files.extend(cat1_1)
    cat1_files.extend(cat1_2)

    cat2_dir = "im_Koilocytotic/CROPPED/"  
    dir1 = os.path.join(cat2_dir,"*.bmp")
    dir = os.path.join(cat2_dir,"*.jpg")
    dir2 = os.path.join(cat2_dir,"*.jpeg")
    cat2_files = glob.glob(dir)
    cat2_files2 = glob.glob(dir2)
    cat2_files1 = glob.glob(dir1)
    cat2_files.extend(cat2_files2)
    cat2_files.extend(cat2_files1)

    cat3_dir = "im_Metaplastic/CROPPED/" 
    dir1 = os.path.join(cat3_dir,"*.bmp")
    dir2 = os.path.join(cat3_dir,"*.jpeg")
    dir = os.path.join(cat3_dir,"*.jpg")
    cat3_files = glob.glob(dir)
    cat3_1 = glob.glob(dir1)
    cat3_2 = glob.glob(dir2)
    cat3_files.extend(cat3_1)
    cat3_files.extend(cat3_2)

    cat4_dir = "im_Parabasal/CROPPED/" 
    dir1 = os.path.join(cat4_dir,"*.bmp")
    dir2 = os.path.join(cat4_dir,"*.jpeg")
    dir = os.path.join(cat4_dir,"*.jpg")
    cat4_files = glob.glob(dir)
    cat4_1 = glob.glob(dir1)
    cat4_2 = glob.glob(dir2)
    cat4_files.extend(cat4_1)
    cat4_files.extend(cat4_2)
    
    cat5_dir = "im_Superficial-Intermediate/CROPPED/" 
    dir1 = os.path.join(cat5_dir,"*.bmp")
    dir2 = os.path.join(cat5_dir,"*.jpeg")
    dir = os.path.join(cat5_dir,"*.jpg")
    cat5_files = glob.glob(dir)
    cat5_1 = glob.glob(dir1)
    cat5_2 = glob.glob(dir2)
    cat5_files.extend(cat5_1)
    cat5_files.extend(cat5_2)
    
    cat1_files.sort()  
    cat2_files.sort()  
    cat3_files.sort()  
    cat4_files.sort() 
    cat5_files.sort()  
    cat1_files=shuffle(cat1_files,random_state=10)
    cat2_files=shuffle(cat2_files,random_state=10)
    cat3_files=shuffle(cat3_files,random_state=10)
    cat4_files=shuffle(cat4_files,random_state=10)
    cat5_files=shuffle(cat5_files,random_state=10)
    
    print("cat1_files:",len(cat1_files))
    print("cat2_files:",len(cat2_files))
    print("cat3_files:",len(cat3_files))
    print("cat4_files:",len(cat4_files))
    print("cat5_files:",len(cat5_files))
    
    total_files=(len(cat1_files)+len(cat2_files)+len(cat3_files)+len(cat4_files))+len(cat5_files)
    
    temp_files=[]
    temp_labels=[]
    for i in range(len(cat3_files)):
        temp_files.append(cat3_files[i])
        temp_labels.append(0)
    
    for i in range(len(cat2_files)):
        temp_files.append(cat2_files[i])
        temp_labels.append(1)
    
    for i in range(len(cat1_files)):
        temp_files.append(cat1_files[i])
        temp_labels.append(2)
        
    for i in range(len(cat4_files)):
        temp_files.append(cat4_files[i])
        temp_labels.append(3)
        
    for i in range(len(cat5_files)):
        temp_files.append(cat5_files[i])
        temp_labels.append(4)
        
    temp_files=shuffle(temp_files,random_state=10)
    temp_labels=shuffle(temp_labels,random_state=10)  
    skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    model_counter=0
    
    early_stopping = EarlyStopping(monitor='val_loss', 
        patience=20, 

        min_delta=0.001, 
        mode='min')
        
       
    model_counter=0    
    accuracy_d=[]
    precision_d=[]
    recall_d=[]
    f_score_d=[]
    
    temp_files=np.array(temp_files)
    temp_labels=np.array(temp_labels)
    

    counter=-1
    my_fold=0
    for train_index, val_index in skf.split(temp_files, temp_labels):
    # Split the data into training and validation sets
        X_train, X_val = temp_files[train_index], temp_files[val_index]
        y_train, y_val = temp_labels[train_index], temp_labels[val_index]
        
        
    
        #train
        train_cat1_files=[]
        for i in range(len(X_train)):
            if y_train[i]==2:
                train_cat1_files.append(X_train[i])

        train_cat2_files=[]
        for i in range(len(X_train)):
            if y_train[i]==1:
                train_cat2_files.append(X_train[i])

        train_cat3_files=[]
        for i in range(len(X_train)):
            if y_train[i]==0:
                train_cat3_files.append(X_train[i])

        train_cat4_files=[]
        for i in range(len(X_train)):
            if y_train[i]==3:
                train_cat4_files.append(X_train[i])
                
        
        train_cat5_files=[]
        for i in range(len(X_train)):
            if y_train[i]==4:
                train_cat5_files.append(X_train[i])
        
        
        #val
        val_cat1_files=[]
        for i in range(len(X_val)):
            if y_val[i]==2:
                val_cat1_files.append(X_val[i])

        val_cat2_files=[]
        for i in range(len(X_val)):
            if y_val[i]==1:
                val_cat2_files.append(X_val[i])

        val_cat3_files=[]
        for i in range(len(X_val)):
            if y_val[i]==0:
                val_cat3_files.append(X_val[i])

        val_cat4_files=[]
        for i in range(len(X_val)):
            if y_val[i]==3:
                val_cat4_files.append(X_val[i])
        
        val_cat5_files=[]
        for i in range(len(X_val)):
            if y_val[i]==4:
                val_cat5_files.append(X_val[i])



    
        train_aug_cat1,train_aug_cat2,train_aug_cat3,train_aug_cat4,train_aug_cat5=no_data_augmentation(train_cat1_files,train_cat2_files,train_cat3_files,train_cat4_files,train_cat5_files)
        val_aug_cat1,val_aug_cat2,val_aug_cat3,val_aug_cat4,val_aug_cat5=no_data_augmentation(val_cat1_files,val_cat2_files,val_cat3_files,val_cat4_files,val_cat5_files)
        

        train_full_data,train_full_label=making_full_data_train(train_aug_cat1,train_aug_cat2,train_aug_cat3,train_aug_cat4,train_aug_cat5)  #getting my full data
        val_full_data,val_full_label=making_full_data(val_aug_cat1,val_aug_cat2,val_aug_cat3,val_aug_cat4,val_aug_cat5)
        
        
        train_full_data,train_full_label= making_training_and_testing_data_train(train_full_data,train_full_label) #dividing full_data into train and test data
        val_full_data,val_full_label=making_training_and_testing_data(val_full_data,val_full_label)
        
        train_data=train_full_data
        test_data=val_full_data
        print("total training data:",len(train_data))
        Model = Sequential()

        pretrained_model= tf.keras.applications.DenseNet169(include_top=False,   
                           input_shape=(224,224,3),
                           classes=5,
                           weights='imagenet')
        for layer in pretrained_model.layers:
                layer.trainable=False

        Model.add(pretrained_model)
        Model.add(Flatten())
        Model.add(Dense(512, activation='relu'))
        Model.add(Dense(5, activation='softmax'))
        Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False), metrics=['accuracy'])
        history=Model.fit(train_data,train_full_label,epochs=100,validation_data=(test_data, val_full_label),batch_size=32,callbacks=[early_stopping])
        pred=Model.predict(test_data)
        predictions = np.argmax(pred,axis = 1)
        y_label=np.argmax(val_full_label,axis = 1)
        print("Accuracy is:",accuracy_score(predictions,y_label))
        accuracy_d.append(accuracy_score(predictions,y_label))
        print("Precision is:",metrics.precision_score(predictions,y_label,average='macro'))
        precision_d.append(metrics.precision_score(predictions,y_label,average='macro'))
        print("Recall is:",metrics.recall_score(predictions,y_label,average='macro'))
        recall_d.append(metrics.recall_score(predictions,y_label,average='macro'))
        print("F1 Score is:",f1_score(predictions, y_label, average='macro'))
        f_score_d.append(f1_score(predictions, y_label, average='macro'))
        print("accuracy of model no. "+str(model_counter)+"is: ",Model.evaluate(val_full_data,val_full_label)[1])
 
        filename ='DenseNet169_model_'+str(model_counter)+'.sav'
                                                                  
        numpy_path='DenseNet169_model_history'+'.npy'
        np.save(numpy_path,history.history)
        pickle.dump(Model, open(filename, 'wb'))
        fold_number=[1,2,3,4,5]
        model_counter+=1
       
    
    zipped = list(zip(fold_number,accuracy_d, precision_d, recall_d, f_score_d))

    df = pd.DataFrame(zipped, columns=['Fold Number','Accuracy', 'Precision', 'Recall', 'F_SCORE'])
    df.to_csv("DenseNet169_model_k_fold_results_with_advance_augmentation_and normal.csv")
    
    