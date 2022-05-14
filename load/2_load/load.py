import numpy as np
import matplotlib.pyplot as plt
import obspy
import csv
from obspy import read
from obspy.taup import TauPyModel
import os
from pathlib import Path
import random
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, ZeroPadding1D
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

count_line = 0

X = [] 
Y = []
X_nst, Y_nev, Z_pro = [], [], []

input_length=1000

### read root of input data
n=0
with open('parameter.list') as rt:
	for line in rt:
		n+=1
		if n==1: nrt=str(line.split()[0])
		if n==2: nmodel=str(line.split()[0])

### read sac data

XKS=['PKS','SKS','SKK']

for k in range(3):
    XKS_rout=nrt+str(XKS[k])+'.list'
    print(XKS_rout)
    with open(XKS_rout) as Pl:
        for line_Pl in Pl:
            vals = line_Pl.split()
            P_rout = nrt+str(vals[0])
            print('Doing: '+str(XKS[k])+' '+str(vals[0]))
            with open(P_rout) as P:
                for line in P:
                    PKS, y = [], []
                    vals = line.split()
                    nst = vals[0]      #station name
                    nev = vals[1]      #event name

                    if vals[13] == 'A' or vals[13] == 'B':
                        y.append(1)
                        y.append(0)
                    else:
                        y.append(0)
                        y.append(1)

                    ncom=['.ro','.to','.rl','.tl']
                    for i in range(4):
                        rtol=[]
                        ro_rout=nrt+str(XKS[k])+'Out/'+str(P_rout[21:26])+'/'+nst+'/'+nev+'/'+nst+str(ncom[i])

                        st=read(ro_rout)
                        if i==0: ro=st[0].data
                        if i==1: to=st[0].data
                        if i==2: rl=st[0].data
                        if i==3: tl=st[0].data

                    for i in range(input_length):
                        PKS.append(np.array([ro[i],to[i],rl[i],tl[i]]))

                    X.append(np.array(PKS))
                    Y.append(np.array(y))
                    X_nst.append(np.array(nst+'_'+str(XKS[k])+'_'))
                    Y_nev.append(np.array(nev))
                    Z_pro.append(np.array(P_rout[21:26]))

            print(np.array(X).shape)

input_shape = (input_length,4)

###start of CNN
model = Sequential()

###add first convolutional layer.
model.add(Conv1D(kernel_size=(3), filters = 32,
          input_shape = input_shape,
          strides = (2),
          activation = 'relu'))
print(model.output_shape)


model.add(ZeroPadding1D(padding = 1))
model.add(Conv1D(kernel_size=(3), filters = 32,
          strides = (2),
          activation = 'relu'))

model.add(ZeroPadding1D(padding = 1))
model.add(Conv1D(kernel_size=(3), filters = 32,
          strides = (2),
          activation = 'relu'))

model.add(Conv1D(kernel_size=(3), filters = 32,
          strides = (2),
          activation = 'relu'))

model.add(ZeroPadding1D(padding = 1))
model.add(Conv1D(kernel_size=(3), filters = 32,
          strides = (2),
          activation = 'relu'))

model.add(Conv1D(kernel_size=(3), filters = 32,
          strides = (2),
          activation = 'relu'))

model.add(Conv1D(kernel_size=(3), filters = 32,
          strides = (2),
          activation = 'relu'))

model.add(ZeroPadding1D(padding = 1))
model.add(Conv1D(kernel_size=(3), filters = 32,
          strides = (2),
          activation = 'relu'))

model.add(ZeroPadding1D(padding = 1))
model.add(Conv1D(kernel_size=(3), filters = 32,
          strides = (2),
          activation = 'relu'))

model.add(Flatten())

###output layer
model.add(Dense(units = (2),
          activation = 'softmax'))
print(model.summary())



print('*-----------------')


###load model
#os.system('/bin/rm -fr *.h5')
model.load_weights(nmodel)


###predict measurement
result = model.predict(np.array(X))


###write result. named as xxxxxx_II_XKS_NC12l_EQxxxxxxx.res
for i in range(len(result)):
        nst=X_nst[i]
        nev=Y_nev[i]
        pro=Z_pro[i]
        res_name='Outp/'+str(nst)+str(pro)+'_'+str(nev)+'.res'
        y_name='Outp/'+str(nst)+str(pro)+'_'+str(nev)+'.y'
        np.savetxt(res_name,result[i])
        np.savetxt(y_name,Y[i])



print('finish')




