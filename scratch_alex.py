from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from scipy.misc import imread, imresize, imshow, imsave
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from sklearn.metrics import confusion_matrix, accuracy_score

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def resize_batch(im_batch,size):
    im_batch_resized=[]
    for im in im_batch:
        im_batch_resized.append(imresize(im,(size,size,3)))
    return np.array(im_batch_resized)

# class PlottingCallback(tflearn.callbacks.Callback):
#     def __init__(self, model, x,
#                  layers_to_observe=(),
#                  kernels=10,
#                  inputs=1):
#         self.model = model
#         self.x = x
#         self.kernels = kernels
#         self.inputs = inputs
#         self.train_losses = []
#         self.observers=[]
#         for l in layers_to_observe:
#             print(tflearn.DNN(l))
#             # self.observers.append(tflearn.DNN(l))
#         self.observers = [tflearn.DNN(l) for l in layers_to_observe]
#     def on_batch_end(self, training_state, snapshot=False):
#         print("The training loss is: ", training_state.global_loss)
#         self.train_losses.append(training_state.global_loss)
#     # def on_epoch_end(self, training_state):
#     #      outputs = [o.predict(self.x) for o in self.observers]
#     #      print([d.shape for d in outputs])#.shape
#          # for i in range(self.inputs):
#          #     plt.figure(frameon=False)
#          #     plt.subplots_adjust(wspace=0.1, hspace=0.1)
#          #     ix = 1
#          #     for o in outputs:
#          #         for kernel in range(self.kernels):
#          #             plt.subplot(len(outputs), self.kernels, ix)
#          #             plt.imshow(o[i, :, :, kernel])
#          #             plt.axis('off')
#          #             ix += 1
#          #     plt.savefig('outputs-for-image:%i-at-epoch:%i.png'
#          #                % (i, training_state.epoch))

def load_info(dirpath):  # image, labels, filename, features
    files = os.listdir(dirpath)
    im = []
    lbl = []
    filenames = []
    features = []
    data_type = dirpath[12:-6]
    feature_array = np.load('../features_' + data_type + '_db.npy')
    filename_array = np.load('../filenames_' + data_type + '_db.npy')
    filename = [str(filename_array[i][43:]) for i in range(filename_array.shape[0])]
    for f in files:
        im.append(imread(dirpath + f))  # find where this filename is in the array
        # we need to get index of dirpath[9:]+f in filename
        idx = filename.index(dirpath[12:] + f)
        lbl.append(f[0])
        features.append(feature_array[idx])
        filenames.append(dirpath + f)
    return im, lbl, filenames, np.asarray(features)

def test_train_split(im,lbl,files,features,ratio=0.25):
    #ratio is num(test samples)/num(train samples)
    total_num=len(files)
    num_train=(int)((1-ratio)*total_num)
    # num_test=total_num-num_train
    shuffle_ids=random.sample(range(total_num),total_num)
    train_ids=shuffle_ids[:num_train]
    test_ids=shuffle_ids[num_train:]
    im1=[im[x] for x in train_ids]
    im2=[im[x] for x in test_ids]
    lbl1=[lbl[x] for x in train_ids]
    lbl2=[lbl[x] for x in test_ids]
    files1=[files[x] for x in train_ids]
    files2=[files[x] for x in test_ids]
    features1=[features[x] for x in train_ids]
    features2=[features[x] for x in test_ids]
    return im1,im2,lbl1,lbl2,files1,files2,features1,features2

def err_calc(im,lbl,model):
    probs=model.predict(im)
    preds=np.argmax(probs,axis=1)
    targets=np.array([int(x) for x in lbl])
    err=1-accuracy_score(targets,preds)
    return err

def calculate_ERE(f1,t1,model,k):
    ere=0
    C=k
    N=len(t1)
    ere_list=[]
    #find probs of all points
    for i in range(len(t1)):
        ere=0
        probs=model.predict(f1[i].reshape(1,50,50,3))[0]
        for c in range(C):
            if c==np.argmax(probs):
                i=0
            else:
                i=1
            ere+=probs[c]*i
        ere_list.append(ere)
    return ere_list

class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self):
        self.train_losses = []
    def on_batch_end(self, training_state, snapshot=False):
        print("The training loss is: ", training_state.global_loss)
        self.train_losses.append(training_state.global_loss)


#loading the data
g_src='../new_data/golden_data/' # golden data path, This is the data we initialize on
s_src='../new_data/seed_data/' #seed path, this is the data, the validation is reported on
d_src='../new_data/mturk_data/' # this is rest of the data, split into test and train

g_im, g_lbl, g_files, g_features = load_info(g_src)
s_im, s_lbl, s_files, s_features = load_info(s_src)
d_im, d_lbl, d_files, d_features = load_info(d_src)

tr_im, test_im, tr_lbl, test_lbl, tr_files, test_files, tr_features, test_features = test_train_split(d_im,d_lbl,d_files,d_features)

#Extract a constant training error measure of 50
shuffle_ids=random.sample(range(len(g_lbl)),50)

g_tr_im=[g_im[x] for x in shuffle_ids]
g_tr_lbl=[g_lbl[x] for x in shuffle_ids]
g_tr_files=[g_files[x] for x in shuffle_ids]
g_tr_features=[g_features[x] for x in shuffle_ids]

# Building 'AlexNet'
#modified for image size
img_aug=ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_blur(sigma_max=3.)
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_flip_updown()
input_data = input_data(shape=[None, 227,227, 3], data_augmentation=img_aug)
conv0 = conv_2d(input_data, 96, 11, strides=4, activation='relu')
max0 = max_pool_2d(conv0, 3, strides=2)
local0 = local_response_normalization(max0)
conv1 = conv_2d(local0, 256, 5, activation='relu')
max1 = max_pool_2d(conv1, 3, strides=2)
local1 = local_response_normalization(max1)
conv2 = conv_2d(local1, 384, 3, activation='relu')
conv3 = conv_2d(conv2, 384, 3, activation='relu')
conv4 = conv_2d(conv3, 256, 3, activation='relu')
max2 = max_pool_2d(conv4, 3, strides=2)
local2 = local_response_normalization(max2)
full1 = fully_connected(local2, 4096, activation='tanh')
drop1 = dropout(full1, 0.5)
full2 = fully_connected(drop1, 4096, activation='tanh',name='layer')
drop2 = dropout(full2, 0.5)
full3 = fully_connected(drop2, 6, activation='softmax')
network = regression(full3, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.001)

n_iter=100
suffix='_scratch_'+str(n_iter)

#initial data used is the golden data
t1=to_categorical(g_lbl,6)
f1=np.array(g_im)

f1=resize_batch(f1,227)

model = tflearn.DNN(network,tensorboard_verbose=3)
model.fit(f1,t1, n_epoch=50, validation_set=0.2, shuffle=True,show_metric=True, batch_size=300, snapshot_step=200, snapshot_epoch=False, run_id='training')
# model.load('model_alex_'+suffix+'.tflearn')

#visualize during test
observed = [input_data, conv1, max1,local1, conv2,conv3, conv4,max2,local2]#, full1,drop1,full2,drop2,full3]
layer_names=['input_data', 'conv0','max0','local0','conv1', 'max1','local1', 'conv2','conv3', 'conv4','max2','local2']#, 'full1','drop1','full2','drop2','full3']
observers = [tflearn.DNN(v, session=model.session) for v in observed]

for i in range(f1.shape[0]):
    outputs = [m.predict(f1[i].reshape(-1,227,227,3)) for m in observers]
    # print([d.shape for d in outputs])
    k=0
    for d in outputs:
        # for i in range(d.shape[3]):
        #print(d, i)
        # imshow(d[0,:,:,i])
        # print(g_files[i],g_files[i][24:])
        idx=random.randint(0,d.shape[3]-1)
        im=np.array(d[0,:,:,idx])
        # print(im.shape,idx)
        name='layers_227/'+g_files[i][24:]+'_'+layer_names[k]+'_'+str(idx)+'.png'
        imsave(name,im) # image number and class label
        k=k+1

# callbacks=PlottingCallback(model, f1[0], (max1, max2),14)
# X_test=f1[0]
# model.fit(f1,t1, n_epoch=5, validation_set=0.2,batch_size=32, callbacks=PlottingCallback(model, f1[0], (max1)))#MonitorCallback())
model.save('model_alex_227_'+suffix+'.tflearn')
#model.load('model_alex_'+suffix+'.tflearn')
