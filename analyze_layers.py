import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.misc import imread

def load_filenames(dirpath):  # image, labels, filename, features
    files = os.listdir(dirpath)
    filenames = []
    for f in files:
        filenames.append(f)
    return filenames

# def split_filenames(filenames):

# get all golden data image names
g_src='../new_data/golden_data/' # golden data path, This is the data we initialize on
g_files= load_filenames(g_src)

g_files0=[]
g_files1=[]
g_files2=[]
g_files3=[]
g_files4=[]
g_files5=[]

for file in g_files:
    # print(file)
    if file[0]=='0':
        g_files0.append(file)
    else:
        if file[0]=='1':
            g_files1.append(file)
        else:
            if file[0]=='2':
                g_files2.append(file)
            else:
                if file[0]=='3':
                    g_files3.append(file)
                else:
                    if file[0]=='4':
                        g_files4.append(file)
                    else:
                        if file[0]=='5':
                            g_files5.append(file)
    # print(g_files0, g_files1, g_files2, g_files3, g_files4, g_files5)

src_files=[]
src_files.append(g_files0)
src_files.append(g_files1)
src_files.append(g_files2)
src_files.append(g_files3)
src_files.append(g_files4)
src_files.append(g_files5)

base_path='layers/'
dirs=os.listdir(base_path)

for src_file in src_files:
    dir=src_file[0][0]
    all_files=os.listdir(base_path+dir)
    for file in src_file:
        imp_files=[]
        base_name=file[:-4]
        for f in all_files:
            if base_name in f:
                imp_files.append(f)
        ordered_imp_files=[]
        order=['input_data', 'conv1', 'max1','local1', 'conv2','conv3', 'conv4','max2','local2']
        for ord in order:
            for f in imp_files:
                if ord in f:
                    ordered_imp_files.append(f)
        fig = plt.figure(figsize=(16, 16))
        plt.subplots_adjust(hspace=0.5, wspace=1.0)
        plt.axis('off')
        columns = 5
        rows = 9
        k = 0
        for i in range(1,len(ordered_imp_files)):
            img = imread(base_path + dir + '/' + ordered_imp_files[k])  # random.randint(10)#, #size=(h,w))
            fig.add_subplot(rows, columns, i)
            # fig.set_title(ordered_imp_files[k][-13:])
            k = k + 1
            plt.imshow(img, cmap='gray')
            for ord in order:
                if ord in ordered_imp_files[k]:
                    plt.title(ord)
        # plt.pause(0.005)
        # plt.show()
        # plt.show()
        plt.savefig('layers_227/layers_' + base_name + '.png')
        # plt.close()
    #for all g_files beginning with 0
#Analyzing all layers

# # w=10
# # h=10
# fig=plt.figure()#figsize=(8, 8))
# columns = 3
# rows = 9
# k=0
# for i in range(1, columns*rows +1):
#     img = imread(base_path+dir+'/'+ordered_imp_files[k])#random.randint(10)#, #size=(h,w))
#     fig.add_subplot(rows, columns, i)
#     k=k+1
#     plt.imshow(img,cmap='gray')
# plt.show()
