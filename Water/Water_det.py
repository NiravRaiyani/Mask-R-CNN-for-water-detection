
"""
Python code for Water detection using Mask R-CNN first proposed by Girshick et al


Created on Tue Jul 30 15:49:57 2019

@author: Nirav Raiyani
"""

# Importing the required Python packages
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


# Get the path of the main file of the project
ROOT_DIR = os.getcwd()

# Getting all the important libraries written for Mask R-CNN (Saved in folder 'mrcnn')

sys.path.append(ROOT_DIR)  # Specifies the path for looking the following packages
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log
import water_p

# Creating the deractory to save logs and weights of the model
MODEL_DIR = os.path.join(ROOT_DIR,"logs")

# Loading the configuration:Object name, No. of epochs and all hyperparameters
config = water_p.WaterConfig() # Configurations are defined in 'water.py' and 'config.py'


# To modify (if needed) some setting in config.
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# Updating the change made in InferenceConfig
config = InferenceConfig()    
config.display()


# Specifying the devise on which Neural Network is to be loaded
DEVICE = "/cpu:0"


# specifying the mode of operation : Inferance or Training
TEST_MODE = "inference"

# Specifying the basic structure for displying the image on matplotlib
#i.e. Array representing the size of the image.

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Creating a model in inferance mode
# tf.device specifies the device to use for operation which in current case is
# our model

with tf.device(DEVICE):
# Callng a MaskRCNN model in 'inferance' mode with aboce specified configurations. 
    model = modellib.MaskRCNN(mode= 'inference', model_dir = MODEL_DIR, config = config)

# Declaring the number of classes available in the model for detection.
class_names = ['BG', 'Water', 'person', 'side_walk']    


    
### Loading the weights of the Model  ###
   
# Specifing the path of the weights
weights_path = "Weight_logs/mask_rcnn_water_0063.h5"

#  Loading the weights
print("Loading the weights of the Mask R-CNN", weights_path)
model.load_weights(weights_path, by_name = True)

# Testing the images with trained image
    
image = cv2.imread('datasets/Water_Person_3/train/Vid-4 image11.jpg')

# Running the image through Mask R-CNN
results = model.detect([image], verbose = 1) 

# Displying the results
ax = get_ax()
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['water_frac'], r['scores'], ax = ax, title = "Water detection results" )   

# def WaterDistrib(r):
#     '''
#     Takes the results given by Mask R-CNN model as an input and generates a
#     scatter plot of water distribution at every pixcel height.
#     '''
#     water_mask=[]
#     for i, j in enumerate(r["class_ids"]):
#         if j == 1:
#             water_mask.append(r["masks"][:,:,i])
    
#     # Converting the bullion mask into binary
#     water_mask = np.squeeze(np.double(water_mask))
    
#     # Summing along the y axis
#     water_sum = np.squeeze( np.sum(water_mask, axis = 1))
        
#     # Getting the co-ordinates of the pixels with mask (along y-axis)
#     co_ord = np.squeeze(np.nonzero(water_sum))

#     # Ploting the scatter plot
#     x = water_sum[co_ord]
#     y = co_ord
    
#     plt.rcdefaults()
#     fig, ax = plt.subplots()
#     ax.scatter(x[0:-1:50], y[0:-1:50], marker = "+")
#     ax.set_xlabel('Spread')
#     ax.set_title('Water distribution')
#     plt.show()
    
    
# WaterDistrib(r)

# ##########################################################################################
# ############### Result Extraction ###############

# water_mask=[]
# for i, j in enumerate(r["class_ids"]):
#     if j == 1:
#         water_mask.append(r["masks"][:,:,i])

# water_mask = np.squeeze(np.double(water_mask))

# # Summing along the y axis
# water_sum = np.squeeze( np.sum(water_mask, axis = 1))

# # Getting the co-ordinates of the pixels with mask (along y-axis)
# co_ord = np.squeeze(np.nonzero(water_sum))


# # Plotting the results
# x = water_sum[co_ord]
# y = co_ord


# plt.rcdefaults()
# fig, ax = plt.subplots()

# # Dividing it into equal space of arsays
# #y_pos = np.arange(co_ord[0], co_ord[0]+len(co_ord))
# #
# #x = np.flipud(x)

# ax.scatter(x[0:-1:50], y[0:-1:50], marker = "+")
# #ax.set_yticks(y_pos,  minor=True)
# ax.set_xlabel('Spread')
# ax.set_title('Water distribution')
# plt.show()




# # Getting the height of the water curtain.
# h_min = np.min(co_ord)
# h_max = np.max(co_ord)
# height = h_max-h_min
# height = height/r['masks'].shape[0]




# def WaterDistrib(r):
#     '''
#     Takes the results given by Mask R-CNN model as an input and generates a
#     scatter plot of water distribution at every pixcel height.
#     '''
#     water_mask=[]
#     for i, j in enumerate(r["class_ids"]):
#         if j == 1:
#             water_mask.append(r["masks"][:,:,i])
    
#     # Converting the bullion mask into binary
#     water_mask = np.squeeze(np.double(water_mask))
    
#     # Summing along the y axis
#     water_sum = np.squeeze( np.sum(water_mask, axis = 1))
        
#     # Getting the co-ordinates of the pixels with mask (along y-axis)
#     co_ord = np.squeeze(np.nonzero(water_sum))

#     # Ploting the scatter plot
#     x = water_sum[co_ord]
#     y = co_ord
    
#     plt.rcdefaults()
#     fig, ax = plt.subplots()
#     ax.scatter(x[0:-1:50], y[0:-1:50], marker = "+")
#     ax.set_xlabel('Spread')
#     ax.set_title('Water distribution')
#     plt.show()

  ###########################here
#m_shape = r['masks'].shape
#k = np.zeros([m_shape[0], m_shape[1], len(class_names) ])


#for i in range(0, len(r['class_ids'])-1):
#    for g in range(0, np.size(k,0)-1):
#        for h in range(0,np.size(k,1)-1):
#            t_co[i] = t_co[i] +1
#            if k[g, h, i]:
#                m_co[i] = m_co[i] + 1
#
#
#j = np.double(r['masks'])
#
#man = np.sum(j[:,:,0])


#def mask_fraction(final_masks):
#    
#    '''
#    Input:[h x w x d] Bullion array of mask of individual object
#          here, d = no. of detected object.
#    Output:[1 x d] Fraction of screen occupied by each object
#    '''
#    k = final_masks
#    
#    # Converting Bullion mask (True/False) to binary(0/1)
#    j = np.double(k, axis = 2)
#    
#    # Summing the pixels with positive mask
#    j = np.sum(j, axis = 0) # sums along row and returns [h x d]
#    j = np.sum(j, axis = 0) # sums along h and returns [d x 1]
#    
#    # Total no. of pixels in the image
#    t = np.multiply(k.shape[0], k.shape[1])
#    
#    # Calculating the fraction of each mask
#    frac = np.devide(j, t)
#    
#    return frac








    
'''
# To get the process flow diagram of the whole process
    
model.keras_model.compile(loss='mean_squared_error', optimizer='sgd')  
from keras.utils import plot_model   
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
plot_model(model.keras_model, to_file='HappyModel.png')
SVG(model_to_dot(model.keras_model).create(prog='dot', format='svg'))    
''' 
    
   
    
    
    
    
    
    
    
    
