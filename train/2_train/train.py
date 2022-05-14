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

### read good and bad measurements sparately
### we will number of enlarge good measurements to balance dataset
X_good, Y_good = [], []
X_bad, Y_bad = [], []
X, Y = [], []
X_rand, Y_rand = [], []


###save name of station, number of event and project name
nst_good, nev_good = [], []
nst_bad, nev_bad = [], []
X_nst, Y_nev = [], []
nst_rand, nev_rand = [], []


### read parameters
n=0
with open('parameters.list') as p:
        for line in p:
                n+=1
                vals = line.split()
                if n==1: nrt=str(vals[0])
                if n==2: batch_size=int(vals[0])
                if n==3: epochs=int(vals[0])
                if n==4: byn=int(vals[0])
                if n==5:
                        ac=int(vals[0])
                        uc=int(vals[1])


input_length = 1000

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

                    if vals[2] == 'A' or vals[2] == 'B':
                        y.append(1)  #label of acceptable measurements
                        y.append(0)
                    else:
                        y.append(0)  #label of unacceptable measurements
                        y.append(1)
                                                
                    ncom=['.ro','.to','.rl','.tl']
                    for i in range(4):
                        rtol=[]
                        ro_rout=nrt+str(XKS[k])+'Out/'+nst+'/'+nev+'/'+nst+str(ncom[i])
                            
                        st=read(ro_rout)
                        if i==0: ro=st[0].data
                        if i==1: to=st[0].data
                        if i==2: rl=st[0].data
                        if i==3: tl=st[0].data
                                
                    for i in range(input_length):
                        PKS.append(np.array([ro[i],to[i],rl[i],tl[i]]))
                    if y[0] == 1:
                        X_good.append(np.array(PKS))
                        Y_good.append(np.array(y))
                        nst_good.append(np.array(nst+'_'+str(XKS[k])+'_'))
                        nev_good.append(np.array(nev))
                    else:
                        X_bad.append(np.array(PKS))
                        Y_bad.append(np.array(y))
                        nst_bad.append(np.array(nst+'_'+str(XKS[k])+'_'))
                        nev_bad.append(np.array(nev))
                            
            print('Accepted: ', np.array(X_good).shape, np.array(Y_good).shape)
            print('Unaccepted: ', np.array(X_bad).shape, np.array(Y_bad).shape)

print(np.array(X_good).shape, np.array(Y_good).shape)
print(np.array(X_bad).shape, np.array(Y_bad).shape)

###enlarge good data based on rate between # of bad and # of good
npts=int(len(X_bad)/len(X_good))
print('Ratio between accpeted and unaccepted measurements: ', npts)

#class_weight
class_weight={0:ac, 1:uc}

#if do not need to balance data
if byn==0: npts=1

#store enlarged good measurements
for i in range(npts):
        for ii in range(int(len(X_good))):
                X.append(np.array(X_good[ii]))
                Y.append(np.array(Y_good[ii]))
                X_nst.append(np.array(nst_good[ii]))
                Y_nev.append(np.array(nev_good[ii]))

#store bad measurement
for i in range(int(len(X_bad))):
        X.append(np.array(X_bad[i]))
        Y.append(np.array(Y_bad[i]))
        X_nst.append(np.array(nst_bad[i]))
        Y_nev.append(np.array(nev_bad[i]))

###random data set
rann0 = random.sample(range(len(X)),len(X))
rann1 = random.sample(range(len(X)),len(X))
rann2 = random.sample(range(len(X)),len(X))
for i in range(len(X)):
        X_rand.append(np.array(X[rann0[rann1[rann2[i]]]]))
        Y_rand.append(np.array(Y[rann0[rann1[rann2[i]]]]))
        nst_rand.append(np.array(X_nst[rann0[rann1[rann2[i]]]]))
        nev_rand.append(np.array(Y_nev[rann0[rann1[rann2[i]]]]))

print("data set shape:",np.array(X_rand).shape,np.array(Y_rand).shape)
print(np.array(nst_rand).shape, np.array(nev_rand).shape)


###resample data for random data set
#90% is train. 10% is validation.
#input shape is x: (n,1000,3), y: (n,2)
x_train, y_train, x_test, y_test = [], [], [], []
x_train = np.array(X_rand[:int(len(X)*0.8)])
y_train = np.array(Y_rand[:int(len(Y)*0.8)])
x_test = np.array(X_rand[int(len(X)*0.8):])
y_test = np.array(Y_rand[int(len(Y)*0.8):])



print(x_train.shape, y_train.shape)


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



###set loss function, learning rate(lr = 0.0001)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

###set input data.
#batch_size means how many data process in one time
#epochs set how many running times for CNN
H=model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = 1,
          class_weight=class_weight,
          validation_data = (x_test, y_test))

np.savetxt('train_64.acc',H.history['acc'])
np.savetxt('train_64.val_acc',H.history['val_acc'])
np.savetxt('train_64.loss',H.history['loss'])
np.savetxt('train_64.val_loss',H.history['val_loss'])

fig1,ax=plt.subplots(1)
plt.plot(H.history['acc'], label='train_acc')
plt.plot(H.history['val_acc'], label='test_acc')
plt.legend()
plt.show()

fig2,ax=plt.subplots(1)
plt.plot(H.history['loss'], label='train_loss')
plt.plot(H.history['val_loss'], label='test_loss')
plt.legend()
plt.show()

###save model
#os.system('/bin/rm -fr *.h5')
model.save_weights('CNN_XKS.h5')

print('Finish')




