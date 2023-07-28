#!/usr/bin/env python
# coding: utf-8

from __future__ import division
import pandas as pd
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.lib.io import file_io
from tensorflow.python.keras.optimizer_v2.adam import Adam
import os
import segmentation_models as sm
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate, Conv2DTranspose, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Input, AvgPool2D
from tensorflow.keras.models import Model
from keras_unet_collection import models
import tensorflow_addons as tfa
import logging
import albumentations as A
from mpl_toolkits.axes_grid1 import make_axes_locatable




gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tensorflow.config.experimental.set_memory_growth(device, True)


# Check GPUS are running

# In[4]:


# gpu_info = get_ipython().getoutput('nvidia-smi')
# gpu_info = '\n'.join(gpu_info)
# if gpu_info.find('failed') >= 0:
#   print('Not connected to a GPU')
# else:
#   print(gpu_info)
# watch -n0.5 nvidia-smi

# from tensorflow.python.client import device_lib
# devices = device_lib.list_local_devices()

# def sizeof_fmt(num, suffix='B'):
#     for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
#         if abs(num) < 1024.0:
#             return "%3.1f %s%s" % (num, unit, suffix)
#         num /= 1024.0
#     return "%.1f%s%s" % (num, 'Yi', suffix)

# for d in devices:
#     t = d.device_type
#     name = d.physical_device_desc
#     l = [item.split(':',1) for item in name.split(", ")]
#     name_attr = dict([x for x in l if len(x)==2])
#     dev = name_attr.get('name', 'Unnamed device')
#     print(f" {d.name} || {dev} || {t} || {sizeof_fmt(d.memory_limit)}")


# Read in the training files

# In[5]:


#read in the trainiing and testing files and get lissts
train_files = pd.read_csv('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_training_files.csv')['Files'].tolist()
val_files = pd.read_csv('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_validation_files.csv')['Files'].tolist()
test_files = pd.read_csv('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_testing_files.csv')['Files'].tolist()


#functin to standardize all bands at once


#function to standardize
def normalize_meanstd(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

#function to normalize
def normalize(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    minv = np.min(a, axis=axis, keepdims=True)
    maxv = np.max(a, axis=axis, keepdims=True)
    return (a - minv) / (maxv - minv)


#function to get files from storage bucket
def get_files(bucket_path):

  """argument is the path to where the numpy
  save files are located, return a list of filenames
  """
  all = []

  #list of files
  files = os.listdir(bucket_path)

  #get list of filenames we will use, notte I remove images that don't have a target due to clouds
  file_names = []
  for f in files:
    if f.endswith('.npy'):


      all.append(os.path.join(bucket_path, f))
  return(all)
  

#get all the pathways
training_data_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_training_data_aug')
training_label_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_training_labels_aug')
validation_data_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_validation_data')
validation_label_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_validation_labels')
testing_data_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_testing_data')
testing_label_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_testing_labels')


print(len(training_data_names))
print(len(validation_data_names))
print(len(testing_data_names))

# t = training_data_names[0].split('/')[:-1]
# t2 = training_data_names[0].split('/')[-1]
# t = '/'.join(t)
# print(t)
# print(t2)


# Lets do 2x augmentation and save all training data to a new folder

# In[ ]:


transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.7),
    A.ShiftScaleRotate(p=0.7),
    A.VerticalFlip(p=0.7)
])

# def my_aug(x_path, y_path, x_out_path, y_out_path, n_aug):
    

#     '''inputs are path to x directory, path to y directory, x_output path, y_output path 
#     and number of times to augment'''
    
#     #make out paths
#     if not os.path.isdir(x_out_path):
#         os.makedirs(x_out_path)
        
#     if not os.path.isdir(y_out_path):
#         os.makedirs(y_out_path)
    
    
#     #loop through files
#     for f in os.listdir(x_path):
    
#         x = np.load(os.path.join(x_path, f))
#         y = np.load(os.path.join(y_path, f))

#         np.save(os.path.join(x_out_path, f), x)
#         np.save(os.path.join(y_out_path, f), y)
        
#         x[x == 0] = -999
#         x[np.isnan(x)] = -999

#         y = y.astype(int)

#         y[y < 0] = 0
#         y[y >1] = 0
#         y[~np.isin(y, [0,1])] = 0

#         y[np.isnan(y)] = 0

#         unique_val = np.unique(y)
        
#         if (np.all(x == -999) == False) and (len(unique_val == 2)):
            
#             #augment two times and keep original image
#             for i in range(n_aug):

#                 transformed = transform(image=x, mask=y)
#                 my_im = transformed['image']
#                 lb = transformed['mask']
                
                
#                 my_im[my_im == 0] = -999
#                 my_im[np.isnan(my_im)] = -999

#                 lb = lb.astype(int)

#                 lb[lb < 0] = 0
#                 lb[lb >1] = 0
#                 lb[~np.isin(lb, [0,1])] = 0

#                 lb[np.isnan(lb)] = 0

#                 unique_val = np.unique(lb)
                
#                 if (np.all(my_im == -999) == False) and (len(unique_val == 2)):
# #                 print(np.nanmax(transformed))
# #                 print(np.nanmax(lb))


#                     i_out = str(i) + '_' + f
#                     np.save(os.path.join(x_out_path, i_out), my_im)
#                     np.save(os.path.join(y_out_path, i_out), lb)

# #run augmentation
# my_aug('/att/nobackup/spotter5/cnn_mapping/nbac_training/ak_training_data', 
#       '/att/nobackup/spotter5/cnn_mapping/nbac_training/ak_training_labels',
#       '/att/nobackup/spotter5/cnn_mapping/nbac_training/ak_training_data_aug', 
#       '/att/nobackup/spotter5/cnn_mapping/nbac_training/ak_training_labels_aug',
#       2)
        
        
        


# Now lets read in the aug file names

# In[9]:


#read in the trainiing and testing files and get lissts
train_files = pd.read_csv('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_training_files.csv')['Files'].tolist()
val_files = pd.read_csv('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_validation_files.csv')['Files'].tolist()
test_files = pd.read_csv('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_testing_files.csv')['Files'].tolist()


#functin to standardize all bands at once


#function to standardize
def normalize_meanstd(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

#function to normalize
def normalize(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    minv = np.min(a, axis=axis, keepdims=True)
    maxv = np.max(a, axis=axis, keepdims=True)
    return (a - minv) / (maxv - minv)


#function to get files from storage bucket
def get_files(bucket_path):

  """argument is the path to where the numpy
  save files are located, return a list of filenames
  """
  all = []

  #list of files
  files = os.listdir(bucket_path)

  #get list of filenames we will use, notte I remove images that don't have a target due to clouds
  file_names = []
  for f in files:
    if f.endswith('.npy'):


      all.append(os.path.join(bucket_path, f))
  return(all)
  

#get all the pathways
training_data_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_training_data_aug')
training_label_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_training_labels_aug')
validation_data_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_validation_data')
validation_label_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_validation_labels')
testing_data_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_testing_data')
testing_label_names = get_files('/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/median_testing_labels')


print(len(training_data_names))
print(len(validation_data_names))
print(len(testing_data_names))


# In[10]:


#for min max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


#for augmentation

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=1.0),
    A.ShiftScaleRotate(p=1.0),
    A.VerticalFlip(p=1.0)
])


# Set up image generator for augmentation

# In[108]:


# #for min max scaling
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()


# #for augmentation

# transform = A.Compose([
#     A.RandomCrop(width=256, height=256),
#     A.HorizontalFlip(p=1.0),
#     A.ShiftScaleRotate(p=1.0),
#     A.VerticalFlip(p=1.0)
# ])

# class img_gen_aug(tensorflow.keras.utils.Sequence):

#     """Helper to iterate over the data (as Numpy arrays).
#     Inputs are batch size, the image size, the input paths (x) and target paths (y)
#     """

#     #will need pre defined variables batch_size, img_size, input_img_paths and target_img_paths
#     def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
#         self.batch_size = batch_size * 3
#         self.img_size = img_size
#         self.input_img_paths = input_img_paths
#         self.target_img_paths = target_img_paths

#     #number of batches the generator is supposed to produceis the length of the paths divided by the batch siize
#     def __len__(self):
#         return len(self.target_img_paths * 3) // self.batch_size #we need to multiply this times three because I am augmenting two images
        
#     def __getitem__(self, idx):
#         """Returns tuple (input, target) correspond to batch #idx."""
#         i = idx * self.batch_size  
#         batch_input_img_paths = self.input_img_paths[i : i + self.batch_size] #for a given index get the input batch pathways (x)
#         batch_target_img_paths = self.target_img_paths[i : i + self.batch_size] #for a given index get the input batch pathways (y)
#         x_stack = np.zeros((self.batch_size,) + self.img_size + (9,), dtype="float32") #create matrix of zeros which will have the dimension (batch_size, height, wideth, n_bands), 8 is the n_bands
#         #do tthe same thing for y
#         y_stack = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        
#         n = 0
       
       
#         #start populating x by enumerating over the input img paths
#         for path in batch_input_img_paths:
                
#                 #get the base path for the training_data
#                 batch_input_img_paths2 = path.split('/')[:-1]
#                 batch_input_img_paths2 = '/'.join(batch_input_img_paths2)
                
#                 #get the file name
#                 f = path.split('/')[-1]
                
#                 #get the base path for the labels
#                 batch_target_img_paths2 = batch_target_img_paths[0].split('/')[:-1]
#                 batch_target_img_paths2 = '/'.join(batch_target_img_paths2)
                

#                 #load x and y
#                 x = np.round(np.load(os.path.join(batch_input_img_paths2, f)), 3)
#                 y = np.load(os.path.join(batch_target_img_paths2, f))

#                 #apply double augmentation
#                 for i in range(2):

#                     #do the augmentation
#                     transformed = transform(image=x, mask=y)
#                     my_im = transformed['image']
#                     lb = transformed['mask']

#                     my_im[my_im == 0] = -999

#                     my_im[np.isnan(my_im)] = -999

#                     my_im[my_im== -999] = np.nan

#                     in_shape = my_im.shape

#                     #turn to dataframe to normalize
#                     my_im = my_im.reshape(my_im.shape[0] * my_im.shape[1], my_im.shape[2])

#                     #normalize 0 to 1
#                     my_im = scaler.fit_transform(my_im)

#                     #convert back to iriginall shape
#                     my_im = my_im.reshape(in_shape)

#                     #replace nan with -1
#                     my_im[np.isnan(my_im)] = -1

#                     #populate x
#                     x_stack[n] = my_im#[:, :, 4:] index number is included, 

#                     #do the same for y
#                     lb = lb.astype(int)

#                     lb[lb < 0] = 0
#                     lb[lb >1] = 0
#                     lb[~np.isin(lb, [0,1])] = 0

#                     lb[np.isnan(lb)] = 0
#                     lb = lb.astype(int)

# #                     print(lb.shape)

#                     # img =  tf.keras.utils.to_categorical(img, num_classes = 2)
#                     y_stack[n] = np.expand_dims(lb, 2) 

#                     n+=1



#                 #now add back in the original data too

#                 x[x == 0] = -999

#                 x[np.isnan(x)] = -999

#                 x[x== -999] = np.nan

#                 in_shape = x.shape

#                 #turn to dataframe to normalize
#                 x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

#                 #normalize 0 to 1
#                 x = scaler.fit_transform(x)

#                 #convert back to iriginall shape
#                 x = x.reshape(in_shape)

#                 #replace nan with -1
#                 x[np.isnan(x)] = -1

#                 #populate x
#                 x_stack[n] = x#[:, :, 4:] index number is included, 

#                 #do the same for y
#                 y = y.astype(int)

#                 y[y < 0] = 0
#                 y[y >1] = 0
#                 y[~np.isin(y, [0,1])] = 0

#                 y[np.isnan(y)] = 0
#                 y = y.astype(int)

#                 y_stack[n] = np.expand_dims(y, 2) 
#                 n+-1            

#         return x_stack, y_stack


# Make generator when I don't want to use augmentation

# In[11]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
class img_gen(tensorflow.keras.utils.Sequence):

    """Helper to iterate over the data (as Numpy arrays).
    Inputs are batch size, the image size, the input paths (x) and target paths (y)
    """

    #will need pre defined variables batch_size, img_size, input_img_paths and target_img_paths
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    #number of batches the generator is supposed to produceis the length of the paths divided by the batch siize
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size] #for a given index get the input batch pathways (x)
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size] #for a given index get the input batch pathways (y)
        x = np.zeros((self.batch_size,) + self.img_size + (9,), dtype="float32") #create matrix of zeros which will have the dimension (batch_size, height, wideth, n_bands), 8 is the n_bands
        
        #start populating x by enumerating over the input img paths
        for j, path in enumerate(batch_input_img_paths):
      
            #load image
            img =  np.round(np.load(path), 3)
           
#             img[img == 0] = -999

#             img[np.isnan(img)] = -999

#             img[img == -999] = np.nan

#             in_shape = img.shape

#             #turn to dataframe to normalize
#             img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

#             #normalize 0 to 1
#             img = scaler.fit_transform(img)

#             #convert back to iriginall shape
#             img = img.reshape(in_shape)

#             #replace nan with -1
#             img[np.isnan(img)] = -1

            #apply standardization
            # img = normalize(img, axis=(0,1))

            #populate x
            x[j] = img#[:, :, 4:] index number is not included, 
            
        #do tthe same thing for y
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        
        for j, path in enumerate(batch_target_img_paths):
                                   
            #load image
            img =  np.round(np.load(path), 3)
               
#             img = img.astype(int)

#             img[img < 0] = 0
#             img[img >1] = 0
#             img[~np.isin(img, [0,1])] = 0
          
#             img[np.isnan(img)] = 0
#             img = img.astype(int)
           
            # img =  tf.keras.utils.to_categorical(img, num_classes = 2)
            y[j] = np.expand_dims(img, 2) 
         

            #Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1

        return x, y


# Read in the images based on the generator

# In[12]:


#batch size and img size
BATCH_SIZE = 15
GPUS = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"]
strategy = tensorflow.distribute.MirroredStrategy() #can add GPUS here to select specific ones
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync



#image size
img_size = (256, 256)

#number of classes to predict
num_classes = 1


#get images
train_gen = img_gen(batch_size, img_size, training_data_names, training_label_names)
val_gen = img_gen(batch_size, img_size, validation_data_names, validation_label_names)
test_gen = img_gen(batch_size, img_size, testing_data_names, testing_label_names)
test_gen_t = img_gen(batch_size, img_size, testing_data_names, testing_label_names)


# Unet model

# Unet ++

# Segmentation moodels Unet

# In[26]:


# model_unet_from_scratch = models.unet_2d((256, 256, 9), filter_num=[64, 128, 256, 512, 1024], 
#                            n_labels=num_classes, 
#                            stack_num_down=2, stack_num_up=2, 
#                            activation='ReLU', 
#                            output_activation='Sigmoid', 
#                            batch_norm=True, pool=True, unpool=True, 
#                            backbone=None, weights=None, 
#                            freeze_backbone=False, freeze_batch_norm=False, 
#                            name='unet')


# In[36]:


help(models.unet_2d)


# In[13]:


# Free up RAM in case the model definition cells were run multiple times
tensorflow.keras.backend.clear_session()
# model = get_model(img_size, num_classes)
# model.summary()


# Train model across multiple GPUS

# In[ ]:


# get_ipython().run_line_magic('load_ext', 'tensorboard')
# #setting learninig rate
# optimizer = tf.keras.optimizers.RMSprop(0.001) #this is 1e-3, default or 'rmsprop'

optimizer = tensorflow.keras.optimizers.Adam() #this is 1e-3, default or 'rmsprop'
LR = 0.0005
    
# optimizer.learning_rate.assign(1e-04) 
# optimizer = tf.keras.optimizers.Adam() 

# optimizer = tf.keras.optimizers.RMSprop(0.001)
loss=tensorflow.keras.losses.BinaryCrossentropy(
    from_logits=False)

# loss=tf.keras.losses.CategoricalCrossentropy(
#     from_logits=False)

#set loss function, optmizer and metric
# model.compile(loss="binary_crossentropy",
#               optimizer='rmsprop',
#               metrics=["accuracy"])
# early_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

# callbacks = [
#     tensorflow.keras.callbacks.ModelCheckpoint("/att/nobackup/spotter5/cnn_mapping/nbac_training/all_36b_no_aug_adam_5_band.h5",
#                                     save_best_only=True)
# ]


callbacks = [tensorflow.keras.callbacks.ModelCheckpoint(
    filepath="/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/15b_adam_dice_9bands_aug_median.h5",
#     verbose=1,
    save_weights_only=False,
    save_best_only=True,
    monitor='val_mean_iou',
    mode = 'max'),
    tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=10)]
    
tensorflow.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', mode = 'min', patience = 10, min_delta=0.001, min_LR = LR/25, verbose = 1)

# callbacks = [tensorflow.keras.callbacks.ModelCheckpoint(
#     filepath="/att/nobackup/spotter5/cnn_mapping/nbac_training/test.h5",
# #     verbose=1,
#     save_weights_only=True,
#     save_best_only=True,
#     monitor='val_mean_iou',
#     mode = 'max'),
#     tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=10)]


# Open a strategy scope.
with strategy.scope():
    
    model_unet_from_scratch = models.unet_2d((256, 256, 9), filter_num=[64, 128, 256, 512, 1024], 
                       n_labels=num_classes, 
                       stack_num_down=2, stack_num_up=2, 
                       activation='ReLU', 
                       output_activation='Sigmoid', 
                       batch_norm=True, pool=False, unpool=False, 
                       backbone='EfficientNetB7', weights=None, 
                       freeze_backbone=False, freeze_batch_norm=False, 
                       name='unet')

#     model_unet_from_scratch = models.unet_2d((256, 256, 9), filter_num=[64, 128, 256, 512, 1024], 
#                        n_labels=num_classes, 
#                        stack_num_down=2, stack_num_up=2, 
#                        activation='ReLU', 
#                        output_activation='Sigmoid', 
#                        batch_norm=True, pool=False, unpool=False, 
#                        backbone='ResNet50', weights='imagenet', 
#                        freeze_backbone=True, freeze_batch_norm=True, 
#                        name='unet')

    model_unet_from_scratch.compile(loss='binary_crossentropy',
                                    optimizer='adam',
                                    metrics=[sm.metrics.Precision(threshold=0.5),
                                      sm.metrics.Recall(threshold=0.5),
                                      sm.metrics.FScore(threshold=0.5), 
                                      sm.metrics.IOUScore(threshold=0.5)])

#fit the model
history = model_unet_from_scratch.fit(
    train_gen,
    epochs=80,
    callbacks = callbacks,
    validation_data=val_gen) 

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')

model_unet_from_scratch.save("/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/15b_adam_dice_9bands_aug_median_backup_120.h5")

history_dict = history.history

#save output
result = pd.DataFrame({'Precision': history_dict["precision"],
                       'Val_Precision': history_dict['val_precision'],
                       'Recall': history_dict["recall"],
                       'Val_Recall': history_dict['recall'],
                       'F1': history_dict["f1-score"],
                       'Val_F1': history_dict['val_f1-score'],
                       'IOU': history_dict["iou_score"],
                       'Val_IOU': history_dict['val_iou_score'],
                       'Loss': history_dict['loss'],
                       'Val_Loss': history_dict['val_loss']})
result.to_csv("/adapt/nobackup/people/spotter5/cnn_mapping/nbac_training/all_15b_no_aug_adam_9_band_aug_median_120.csv")




