#License Terms

#Copyright (c) 2020-21, California Institute of Technology ("Caltech").  U.S. Government sponsorship acknowledged.

#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#* Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#* Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



# Dependencies
import numpy as np
from matplotlib import pyplot as plt
from libtiff import TIFF
import scipy as sp
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, UpSampling2D, Concatenate, Add, Multiply, LeakyReLU, BatchNormalization, Dropout
from scipy.interpolate import RectBivariateSpline
from io import StringIO
import multiprocessing

def rescale(img, scale):
    """
    Downscales an image by the factor given in parameter 'scale', which should be a power of 2.
    The dimensions of the image should be an integer multiple of scale.
    """
    sh = list(img.shape)
    result = np.zeros((sh[0] // scale, sh[1] // scale))
    for i in range(0, sh[0], scale):
        for j in range(0, sh[1], scale):
            result[i//scale,j//scale] = np.average(img[i:i+scale,j:j+scale])
    return result

def rescale_generic(img_input, scale):
    """
    Downscales an image by an arbitrary integer factor given in parameter 'scale'.
    The dimensions of the image do not need to be an integer multiple of scale, in which case the image
    will be extended along the right and bottom edges.
    """
    sh = list(img_input.shape)
    rsh = (int(np.ceil(sh[0] / scale)), int(np.ceil(sh[1] / scale)))
    if (rsh[0] * scale > sh[0] or rsh[1] * scale > sh[1]):
        img = np.zeros((rsh[0] * scale, rsh[1] * scale), img_input.dtype)
        img[:sh[0],:sh[1]] = img_input
        for i in range(0, (sh[0] // scale) + 1):
            img[i*scale:(i+1)*scale,sh[1]:rsh[1]*scale] = np.average(img_input[i*scale:(i+1)*scale,(rsh[1] - 1)*scale:sh[1]])
        for i in range(0, (sh[1] // scale) + 1):
            img[sh[0]:rsh[0]*scale,i*scale:(i+1)*scale,] = np.average(img_input[(rsh[0] - 1)*scale:sh[0],i*scale:(i+1)*scale])
        img[(rsh[0] - 1)*scale:rsh[0]*scale,(rsh[1] - 1)*scale:rsh[1]*scale] = np.average(img_input[(rsh[0] - 1)*scale:sh[0],(rsh[1] - 1)*scale:sh[1]])
        img[(rsh[0] - 1)*scale:sh[0],(rsh[1] - 1)*scale:sh[1]] = img_input[(rsh[0] - 1)*scale:sh[0],(rsh[1] - 1)*scale:sh[1]]
    else:
        img = img_input

    # https://scipython.com/blog/binning-a-2d-array-in-numpy/
    shape = (rsh[0], scale,
             rsh[1], scale)
    return img.reshape(shape).mean(-1).mean(1)

def findBestEnhanceModel(lat,lon):
    """Find all models in the model dir, pick the one on the closest latitude, then longitude."""
    dir = "../models/"
    allModels=os.listdir(dir)
    geoModels=[allModels[i] for i in range(len(allModels)) if allModels[i][:14]=="model_enhance_"]
    allTilesLL = np.array([np.genfromtxt(StringIO(gm[14:]),delimiter="_") for gm in geoModels])
    dtype = [('lat dist',float),('lon dist',float),('lat',float),('lon',float)]
    allTilesLLMetric = np.array([(np.abs(v[0]-lat),np.abs(v[1]-lon),v[0],v[1]) for v in allTilesLL],dtype=dtype)
    bestModel=np.sort(allTilesLLMetric,order=['lat dist','lon dist'])[0]
    print("[ LOG ] best model at (" + str(bestModel[2]) + "," + str(bestModel[3]) + "), distance (" + \
          str(bestModel[0]) + "," + str(bestModel[1]) + ")")
    bestModel2=np.array([bestModel[i] for i in range(2,4)]).astype(np.int)
    fileName = dir+"model_enhance_"+str(bestModel2[0])+"_"+str(bestModel2[1])
    return fileName

def rescaleDown(lims,data):
    return np.interp(data,lims,(0,1))

def rescaleUp(lims,data):
    return np.interp(data,(0,1),lims)

def openDem(path):
    print("[ LOG ] Opening DEM... " + path)
    return TIFF.open(path).read_image()

def getMola(latu,latl,lonl,lonr,mola):
    # Using a slight offset to get better alignment with CTX images, resulting in dramatic improvement
    m = mola[128*(90-latu)+1:128*(90-latl)+1,128*(180+lonl)+2:128*(180+lonr)+2]
    if m.shape[1] == 510:
        new = np.zeros((m.shape[0], 512))
        new[:,:510] = m[:,:]
        new[:,510] = mola[128*(90-latu)+1:128*(90-latl)+1,0]
        new[:,511] = mola[128*(90-latu)+1:128*(90-latl)+1,1]
        m = new
    if m.shape[0] == 511:
        new = np.zeros((512, 512))
        new[:511,:] = m[:,:]
        new[511,:] = m[510,:]
        m = new
    return m.astype(np.int16)

def getCTXFilename(prefix, lat, lon):
    return prefix + "Lab_CTX-Mosaic_beta01_E"+str(lon).zfill(3+int(0.5-0.5*np.sign(lon)))+ \
           "_N"+str(lat).zfill(2+int(0.5-0.5*np.sign(lat)))+".tif"

def getCTX(lat,lon):
    # return a 2 degree square tile of CTX data referenced by lat and lon.
    dire = "../data/zips/"
    file = getCTXFilename("Murray-", lat, lon)
    if not os.path.isfile(dire+file):
        # Inconsistencies in the naming...
        file = getCTXFilename("Murray", lat, lon)
    try:
        img = TIFF.open(dire+file).read_image()
        # Some of the 2x2 tiles have missing data along the edges. Simply copy the adjacent row/column
        if img[1,0] == 0 and img[2,0] == 0 and img[3,0] == 0 and img[1,1] > 0 and img[2,1] > 0 and img[3,1] > 0:
            x = img[:,1]
            img[:,0] = x
        if img[0,1] == 0 and img[0,2] == 0 and img[0,3] == 0 and img[1,1] > 0 and img[1,2] > 0 and img[1,3] > 0:
            x = img[1, :]
            img[0, :] = x
        # Fill in areas with missing data to avoid creating sharp edges
        # Don't do this when training the enhance model!
        img[img == 0] = 128
        return img
    except:
        print("ERROR: Could not find "+dire+file+", inserted zeros.")
        return np.zeros((23710,23710))

def saveDem(path,dem,split=False):
    if not split:
        # 32bit precision is plenty
        converted = dem.astype(np.float32)
        tif = TIFF.open(path+".tif", mode='w')
        tif.write_image(converted)
    else:
        sf = int(np.ceil(dem.shape[0]/8192))
        size = int(dem.shape[0]/sf)
        for i in range(sf):
            for j in range(sf):
                tif = TIFF.open(path+"_"+str(i)+"_"+str(j)+".tif", mode='w')
                tif.write_image(dem[i*size:(i+1)*size,j*size:(j+1)*size])

def ctxS(data):
    return rescaleDown((0,255),data)

def getInterpolatedCTX(lats,lons,ctx,latlims,lonlims):
    eps = 1e-6
    if np.min(lats)<latlims[0]-eps or np.max(lats)>latlims[1]+eps or np.min(lons)<lonlims[0]-eps or np.max(lons)>lonlims[1]+eps:
        return "Error, selected area must be between "+str(latlims)+" and E"+str(lonlims)+"."
    # Work out which part of ctx we want and at what resolution
    # lim# represents the extent of the ctx sub-space selection
    lim1 = np.maximum(0, int(np.floor((np.min(lons) - lonlims[0]) / (lonlims[1] - lonlims[0]) * ctx.shape[1])))
    lim2 = int(np.ceil((np.max(lons)-lonlims[0])/(lonlims[1]-lonlims[0])*ctx.shape[1]))
    lim3 = np.maximum(0,int(np.floor((latlims[1]-np.max(lats))/(latlims[1]-latlims[0])*ctx.shape[0])))-1
    lim4 = int(np.ceil((latlims[1]-np.min(lats))/(latlims[1]-latlims[0])*ctx.shape[0]))-1
    # Calculate stride.
    lonStride = int(np.maximum(1,np.floor(0.5*(lim2-lim1)/(len(lons)-1))))
    latStride = int(np.maximum(1,np.floor(0.5*(lim4-lim3)/(len(lats)-1))))
    
    # Now calculate the actual lats and lons represented by this sampling
    lonInd = np.arange(lim1,lim2+1,lonStride)
    latInd = np.arange(lim3,lim4+1,latStride)
    lons2 = lonInd/ctx.shape[1]*(lonlims[1]-lonlims[0])+lonlims[0]
    lats2 = np.flip(latlims[1]-latInd/ctx.shape[0]*(latlims[1]-latlims[0]))
    
    # Calculate interpolator
    interp = RectBivariateSpline(lats2, lons2, rescale_generic(ctx[lim3:lim4+1,lim1:lim2+1], lonStride))
    return interp(lats,lons)

def getInterpolatedCTXBigOptRow(args):
    (trigger,split,i,lats,lons,ctx,latlims,lonlims) = args
    result = np.zeros((trigger, len(lons)))
    for j in range(split):
        # Take care of how latitude is represented from bottom up
        result[0:trigger,j*trigger:(j+1)*trigger] = getInterpolatedCTX(lats[(split-1-i)*trigger:(split-1-i+1)*trigger],
                                        lons[j*trigger:(j+1)*trigger],
                                        ctx,
                                        latlims,#ctx, latlims, lonlims are a matched triplet.
                                        lonlims)
    return result

def getInterpolatedCTXBigOpt(pool,lats,lons,ctx,latlims,lonlims):
    # In case of any problems, lower this to 512
    trigger = 2048

    if len(lats) < trigger+1:
        return getInterpolatedCTX(lats,lons,ctx,latlims,lonlims)
    else:
        split = int(len(lats)/trigger) # Expect this to be an integer power of 2
        output = np.zeros((len(lats),len(lons)))
        args = []
        for i in range(split):
            args.append([trigger, split, i, lats, lons, ctx, latlims, lonlims])
        
        # There is a slight advantage in running this parallel especially when enhancing beyond 115m
        results = pool.map(getInterpolatedCTXBigOptRow, args)

        for i in range(split):
            # Take care of how latitude is represented from bottom up
            output[i*trigger:(i+1)*trigger,:] = results[i]

        return output


def getInterpolatedMOLA(lats,lons,mola,latlims,lonlims):
    mol = getMola(latlims[1],latlims[0],lonlims[0],lonlims[1],mola)
    eps = 1e-6
    if np.min(lats)<latlims[0]-eps or np.max(lats)>latlims[1]+eps or np.min(lons)<lonlims[0]-eps or np.max(lons)>lonlims[1]+eps:
        return "Error, selected area must be between "+str(latlims)+" and E"+str(lonlims)+"."
    # Work out which part of ctx/mola we want and at what resolution
    # lim# represents the extent of the ctx/mola sub-space selection
    lim1 = np.maximum(0,int(np.floor((np.min(lons)-lonlims[0])/(lonlims[1]-lonlims[0])*mol.shape[1])))
    lim2 = int(np.ceil((np.max(lons)-lonlims[0])/(lonlims[1]-lonlims[0])*mol.shape[1]))
    lim3 = np.maximum(0,int(np.floor((latlims[1]-np.max(lats))/(latlims[1]-latlims[0])*mol.shape[0])))-1
    lim4 = int(np.ceil((latlims[1]-np.min(lats))/(latlims[1]-latlims[0])*mol.shape[0]))-1
    # Calculate stride.
    lonStride = int(np.maximum(1,np.floor(0.5*(lim2-lim1)/(len(lons)-1))))
    latStride = int(np.maximum(1,np.floor(0.5*(lim4-lim3)/(len(lats)-1))))
    
    # Now calculate the actual lats and lons represented by this sampling
    lonInd = np.arange(lim1,lim2+1,lonStride)
    latInd = np.arange(lim3,lim4+1,latStride)
    lons2 = lonInd/mol.shape[1]*(lonlims[1]-lonlims[0])+lonlims[0]
    lats2 = np.flip(latlims[1]-latInd/mol.shape[0]*(latlims[1]-latlims[0]))
    
    # Calculate interpolator
    interp = RectBivariateSpline(lats2, lons2, mol[lim3:lim4+1:latStride,lim1:lim2+1:lonStride])
    return interp(lats,lons)

def interpPiece(model,im,dem,rescale_lims,dim,enhance=False):
    # im has twice resolution of dem, matches model trained shape
    inp = [np.zeros((1,2*dim,2*dim,1)),np.zeros((1,dim,dim,1))]
    inp[0][0,:,:,0] = ctxS(im)
    inp[1][0,:,:,0] = rescaleDown(rescale_lims,dem)
    if enhance:
        return rescaleUp(rescale_lims,model.predict(inp[0])[0,:,:,0])
    else:
        return rescaleUp(rescale_lims,model.predict(inp[1])[0,:,:,0])
    # Not sure why this doesn't work well, but the resulting diff is very blocky regardless of
    # interpolation order. Reverted to imperfect deep learning interpolation.
    # Spline interpolation rather than deep learning-based.
        #return RectBivariateSpline(np.arange(0,2*dim,2),
        #                           np.arange(0,2*dim,2),
        #                           dem,kx=4,ky=4)(np.arange(0,2*dim,1), np.arange(0,2*dim,1))

def interpBigImg(model,ctx,mola,rescale_lims,padding,dim,enhance=False):
    """ For better GPU utilisation calling predict on batches """
    sT = time.time()
    csh = list(ctx.shape)
    # Pad output by the useless margins.
    upad = padding[0]
    pad = padding[1]
    csh[0]+=pad+upad
    csh[1]+=pad+upad
    # Construct inputs
    ctxInput = np.zeros(csh)
    ctxInput[upad:-pad,upad:-pad]=ctx
    for i in range(pad):
        ctxInput[-pad+i] = ctxInput[-pad-1+i]
        ctxInput[:,-pad+i] = ctxInput[:,-pad-1+i]
    for i in range(upad):
        ctxInput[upad-i-1] = ctxInput[upad-i]
        ctxInput[:,upad-i-1] = ctxInput[:,upad-i]
    # Expand MOLA
    molaInput = np.zeros(csh)
    molaInput[upad:-pad:2,upad:-pad:2] = mola
    hpad = int(pad/2)
    for i in range(hpad):
        molaInput[::2,::2][-hpad+i] = molaInput[::2,::2][-hpad-1+i]
        molaInput[::2,::2][:,-hpad+i] = molaInput[::2,::2][:,-hpad-1+i]
    uhpad = int(upad/2)
    for i in range(uhpad):
        molaInput[::2,::2][uhpad-i-1] = molaInput[::2,::2][uhpad-i]
        molaInput[::2,::2][:,uhpad-i-1] = molaInput[::2,::2][:,uhpad-i]
    # Specify output container
    output = np.zeros(csh)
    # fill in, then do right and bottom edges.
    # dim is the size of the model mola input square, ie half of the ctx input.
    stride = 2*dim-pad-upad

    num_samples = (csh[1] - 2 * dim) // stride + 1

    for i in range(0,csh[0]-2*dim,stride):    
        if enhance:
            batch = np.zeros((num_samples, 2 * dim, 2 * dim, 1))
            for j in range(0,csh[1]-2*dim,stride):
                batch[j//stride,:,:,0] = ctxS(ctxInput[i:i+2*dim,j:j+2*dim])
        else:
            batch = np.zeros((num_samples, dim, dim, 1))
            for j in range(0,csh[1]-2*dim,stride):
                batch[j//stride,:,:,0] = rescaleDown(rescale_lims, molaInput[i:i+2*dim:2,j:j+2*dim:2])
        res = model.predict_on_batch(batch)
        for j in range(0,csh[1]-2*dim,stride):
            output[i:i+2*dim,j:j+2*dim][upad:-pad,upad:-pad] = rescaleUp(rescale_lims, res[j//stride,:,:,0])[upad:-pad,upad:-pad]

    
    if enhance:
        batch = np.zeros((2, num_samples, 2 * dim, 2 * dim, 1))
        for i in range(0,csh[0]-2*dim,stride):
            batch[0,i//stride,:,:,0] = ctxS(ctxInput[i:i+2*dim,-2*dim:])
            batch[1,i//stride,:,:,0] = ctxS(ctxInput[-2*dim:,i:i+2*dim])
    else:
        batch = np.zeros((2, num_samples, dim, dim, 1))
        for i in range(0,csh[1]-2*dim,stride):
            batch[0,i//stride,:,:,0] = rescaleDown(rescale_lims, molaInput[i:i+2*dim:2,-2*dim::2])
            batch[1,i//stride,:,:,0] = rescaleDown(rescale_lims, molaInput[-2*dim::2,i:i+2*dim:2])
    res0 = model.predict_on_batch(batch[0,:,:,:,0])
    res1 = model.predict_on_batch(batch[1,:,:,:,0])
    for i in range(0,csh[0]-2*dim,stride):
        output[i:i+2*dim,-2*dim:][upad:-pad,upad:-pad] = rescaleUp(rescale_lims, res0[i//stride,:,:,0])[upad:-pad,upad:-pad]
        output[-2*dim:,i:i+2*dim][upad:-pad,upad:-pad] = rescaleUp(rescale_lims, res1[i//stride,:,:,0])[upad:-pad,upad:-pad]

    output[-2*dim:,-2*dim:][upad:-pad,upad:-pad] = interpPiece(model, ctxInput[-2*dim:,-2*dim:], molaInput[-2*dim::2,-2*dim::2],
                                                               rescale_lims, dim, enhance)[upad:-pad,upad:-pad]
    print("[ LOG ] interpBigImg " + str(int(time.time() - sT)) + " sec")
    return output[upad:-pad,upad:-pad]

# This is a bit ugly because originally there was a single neural network and now interpolation and correction
# are done separately.

def getTrainingData(index,size,molaTrain,ctxTrain):
    molaTrainScaled = rescale(molaTrain[index[0]:index[0]+size,index[1]:index[1]+size], 2)
    return [ctxTrain[index[0]:index[0]+size,index[1]:index[1]+size],
            molaTrainScaled,
            molaTrain[index[0]:index[0]+size,index[1]:index[1]+size]]

def getTrainingDataCTX(index,size,molaTrain,ctxTrain,molaTrainInterp):
    molaTrainScaled = rescale(molaTrain[index[0]:index[0]+size,index[1]:index[1]+size], 2)
    return [ctxTrain[index[0]:index[0]+size,index[1]:index[1]+size],
            molaTrainScaled,
            (molaTrain-molaTrainInterp)[index[0]:index[0]+size,index[1]:index[1]+size]]

# Generate neural nets

def generateInterp(dim):
    # Specify neural net
    input2 = Input(shape=(dim,dim,1))
    x2 = UpSampling2D(size=2, interpolation='bilinear')(input2)
    x2 = Conv2D(2*dim**2,4,padding='same',activation='relu')(x2)
    x2 = Conv2D(1,4,padding='same',activation='linear')(x2)
    out = x2
    model = Model(inputs=input2, outputs = out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def trainInterp(model,dim,demRes,molaLims,molaTrain,ctxTrain):
    # Generate training data
    trainingSet = [getTrainingData(np.random.randint(0,high=demRes-2*dim,size=2),
                                   2*dim,molaTrain,ctxTrain) for x in range(3200)]
    lTS = len(trainingSet)
    X2 = np.zeros((lTS,dim,dim,1))
    y = np.zeros((lTS,2*dim,2*dim,1))
    for i in range(lTS):
        X2[i,:,:,0] = rescaleDown(molaLims,trainingSet[i][1])
        y[i,:,:,0] = rescaleDown(molaLims,trainingSet[i][2])
    # train
    # 10 epochs is heaps. Overfitting can be a real problem here.
    model.fit(X2,y, epochs=10, batch_size=8,validation_split=0.2)
    # Evaluate
    test = model.evaluate(X2,y, batch_size=8)
    if test<10**-5:
        print("[ LOG ] Interpolator probably okay.")
    else:
        print("[ LOG ] Interpolator loss above 10^-5, possibly not good enough.")
    # Save
    model.save('../models/model_interp_'+str(lat)+"_"+str(lon))
    # Later rename the good one "interp1" to be generically used.
    return model

# More generally it turns out that not all 4 degree square graticules are created equal.
# Some are just rubbish for training at the requisite scale, while Gale turns out to be quite good.

def generateEnhance(dim):
    # Specify neural net
    # Based on pix2pix https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py

    # bulk up interior layers
    fac = 1 # No actual improvement sighted from doing this.
    
    input1 = Input(shape=(2*dim,2*dim,1))#32 (size of array in width)
    # Downsampling
    x0 = Conv2D(1*fac*dim**2,4,padding='same',activation='relu',strides=2)(input1)#16
    x1 = Conv2D(2*fac*dim**2,4,padding='same',activation='relu',strides=2)(x0)#8
    x2 = Conv2D(4*fac*dim**2,4,padding='same',activation='relu',strides=2)(x1)#4

    # Upsampling
    x5 = UpSampling2D(size=2)(x2)#8
    x5 = Conv2D(2*fac*dim**2,4,padding='same',activation='relu',strides=1)(x5)
    x5 = Concatenate()([x5,x1])
    x6 = UpSampling2D(size=2)(x5)#16
    x6 = Conv2D(1*fac*dim**2,4,padding='same',activation='relu',strides=1)(x6)
    x6 = Concatenate()([x6,x0])

    #Output
    out = UpSampling2D(size=2)(x6)#32
    out = Conv2D(1,4,padding='same',activation='linear',strides=1)(out)

    model = Model(inputs=input1, outputs = out)
    # All extremely generic and unexciting.
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def trainEnhance(model,dim,demRes,mola2lims,molaTrain,ctxTrain,molaTrainInterp,epochs,lat,lon):
    trainingSet = [getTrainingDataCTX(np.random.randint(0,high=demRes-2*dim,size=2),2*dim,
                                      molaTrain,ctxTrain,molaTrainInterp) for x in range(3200)]
    # Generate training data
    lTS = len(trainingSet)
    X1 = np.zeros((lTS,2*dim,2*dim,1))
    y = np.zeros((lTS,2*dim,2*dim,1))
    for i in range(lTS):
        X1[i,:,:,0] = ctxS(trainingSet[i][0])
        y[i,:,:,0] = rescaleDown(mola2lims,trainingSet[i][2])
    # Train
    # Adjust learning rate (turns out to not be very useful).
    #def lr_decay(epoch):
    #    return 0.001*0.001**(epoch/epochs)

    #lr schedule callback
    #lr_decay_callback = keras.callbacks.LearningRateScheduler(lr_decay, verbose=False)

    #opt = keras.optimizers.Adam(learning_rate=0.0001) # optimizer=opt # needs to go in compile
    model.fit(X1,y, epochs=epochs, batch_size=16, validation_split=0.2)#, callbacks=[lr_decay_callback], verbose=1)

    # At the end of training there's some thing causing a huge waste of time, and it's getting worse.
    # Probably due to driver problems but I don't want to touch that...
    
    # Evaluate
    print("[ LOG ] Evaluating model")
    test = model.evaluate(X1[-640:],y[-640:], batch_size=16)
    if test<1e-3:
        print("[ LOG ] Enhancer probably okay.")
    else:
        print("[ LOG ] Enhancer loss above 1e-3, possibly not good enough.")
    # Save
    model.save('../models/model_enhance_'+str(lat)+"_"+str(lon))
    print("[ LOG ] returning model")
    return model

def enhanceMain(pool, mola, lat, lon, res, model_enhance = None):
    # defines 4 square degree graticule of given planet defined by lat, lon. Top left corner. 
    # Enhances until resolution better than res. User expected to know limits of imagery, around 5 m for CTX. 
    tileSize=4 # 4 degrees x 4 degree tiles, corresponding to Murray lab CTX data
    dim=16 # dimension of origin DEM tile, image and output are 2*dim.
    interp_dim = dim
    planetRadius=3389500 #m
    
    print("[ LOG ] planetary DEM enhancement tool operating on " + str(tileSize) + " degree tile from")
    print("[ LOG ] latitude: [" + str(lat) + "," + str(lat-tileSize) + "], longitude: [" + str(lon) + "," + str(lon+tileSize)+"].")
    
    initialRes = 2*np.pi*3389500/mola.shape[1]
    print("[ LOG ] Initial DEM resolution "+str(initialRes)+" m.")
    numEnhance = int(np.ceil(np.log2(462/res)))
    print("[ LOG ] Will enhance " + str(numEnhance)+" time(s), a total resolution increase of "+str(2**numEnhance)+" to " +
          str(initialRes/2**numEnhance) + " m.")

    # Get ctx
    print("[ LOG ] Opening CTX...")
    ctx = np.concatenate([
            np.concatenate([
                getCTX(lat,lon)
            for lat in range(lat-2,lat-2-tileSize,-2)],axis=0)
          for lon in range(lon,lon+tileSize,2)],axis=1)

    # Build training set
    print("[ LOG ] Building training data tiles...")
    demRes = getMola(lat,lat-tileSize,lon,lon+tileSize,mola).shape[0]
    ctxTrain=getInterpolatedCTX(np.arange(lat-tileSize,lat,tileSize/demRes),
                                np.arange(lon,lon+tileSize,tileSize/demRes),
                                ctx,
                                [lat-tileSize,lat],
                                [lon,lon+tileSize])
    molaTrain = getInterpolatedMOLA(np.arange(lat-tileSize,lat,tileSize/demRes),
                                    np.arange(lon,lon+tileSize,tileSize/demRes),
                                    mola,
                                    [lat-tileSize,lat],
                                    [lon,lon+tileSize])

    print("[ LOG ] Initializing scaling helper functions.")
    # Rescale data to within 0,1 for machine learning stuff. Leave some more room for peaks. Relatively flat tiles can shift quite a bit.
    molaMin=np.min(molaTrain)
    molaMax=np.max(molaTrain)
    molaLims=(1.04*molaMin-0.04*molaMax,1.04*molaMax-0.04*molaMin)

    # Get models (search for nearby ones, build dir, or start from scratch)
    print("[ LOG ] Loading generic interp model.")
    try:
        model_interp = keras.models.load_model('../models/model_interp_'+str(lat)+"_"+str(lon))
        print("[ LOG ] Loaded interp model.")
    except:
        print("[ LOG ] No interp model found, building one from scratch...")
        print("[ LOG ] Generating interp model.")
        # generate interp model
        model_interp = generateInterp(interp_dim)
        print(model_interp.summary())
        # train interp
        model_interp = trainInterp(model_interp,interp_dim,demRes,molaLims,molaTrain,ctxTrain)

    print("[ LOG ] Generating interp error tile...")
    #interpPadding=[interp_dim // 3, interp_dim // 3]
    interpPadding=[6, 8]
    molaTrainInterp = interpBigImg(model_interp, ctxTrain, rescale(molaTrain, 2), molaLims, interpPadding, interp_dim, False)
    # Keep an eye on how pathological the corrections are. Hopefully they're smoothish.
    
    # Add to scaling functions
    mola2Min=np.min(molaTrain-molaTrainInterp)
    mola2Max=np.max(molaTrain-molaTrainInterp)
    mola2lims=(mola2Min,mola2Max)
    
    if model_enhance is None:
        try:
            print("[ LOG ] Loading closest enhance model.")
            # Build directory of enhance models.
            bestModel = findBestEnhanceModel(lat,lon)
            model_enhance = keras.models.load_model(bestModel)
            print("[ LOG ] Loaded generic enhance model.")
        except:
            print("[ LOG ] No enhance model found, building from scratch...")
            # generate enhance model
            model_enhance = generateEnhance(dim)
            # train enhance model
            model_enhance = trainEnhance(model_enhance,dim,demRes,mola2lims,molaTrain,ctxTrain,molaTrainInterp,50,lat,lon)
        print(model_enhance.summary())

    # It turns out that localizing probably doesn't help much, at least away from the poles.
    # Train enhance model only, maybe a bit less than before (only 10 epochs)
    #print("Localizing enhance model for this tile...")
    #model_enhance = trainEnhance(model_enhance,dim,demRes,mola2lims,molaTrain,ctxTrain,molaTrainInterp,100,lat,lon)
    
    # Define enhance function using models
    enhancePadding=interpPadding
    def enhanceFunction(ctxInput,molaInput):
        return interpBigImg(model_enhance,ctxInput,molaInput,mola2lims,enhancePadding,dim,True) + \
               interpBigImg(model_interp,ctxInput,molaInput,molaLims,interpPadding,interp_dim,False)

    # There's a bug in the rectbilinearspline function that is called here, for exceptionally
    # large arrays, it would be addressed by chunking it. Want to avoid edge effects, so need some overlap.
    def getEnhancedCTX(factor):
        return getInterpolatedCTXBigOpt(pool, np.arange(lat-tileSize,lat,tileSize/(factor*demRes)),
                                     np.arange(lon,lon+tileSize,tileSize/(factor*demRes)),
                                     ctx,
                                     [lat-tileSize,lat],
                                     [lon,lon+tileSize])

    # Enhance until resolution condition met
    sT = time.time()
    enhancedDem = molaTrain
    for i in range(numEnhance):
        currentResolution = initialRes/2**(i+1)
        print("[ LOG ] Enhancement "+str(i+1)+" of "+str(numEnhance)+".")
        print("[ LOG ] Current resolution " + str(currentResolution)+ " m.")
        enhancedCTX = getEnhancedCTX(2**(i+1))
        enhancedDem = enhanceFunction(enhancedCTX, enhancedDem)
    
    # Saving only the end result
    print("[ LOG ] Saving enhanced DEM")
    saveDem("../output/enhanced_dem_"+str(lat)+"_"+str(lon)+"_"+str(int(currentResolution))+"_m",enhancedDem,False)

    print("[ LOG ] Enhance took " + str(int(time.time() - sT)) + " sec")


#for lon in range(132,-182,-4):
#    enhanceMain(-4,lon,150)

# Set latitude band, lat at top of band
# Gale (-4, 136)
# Jezero (20, 76)
#lat = 20 # degrees latitude, top of 4 degree band
#lat = -4

#for lati in range(8,91,4):
#    for si in range(-1,2,2):
#        lat=si*lati


#
# Usage:
#    ctxEnhanceScript.py <latitude> <longitude_start> <longitude_end>
#
# 32GB RAM is recommended to run this due to the unoptimized rescale_generic function.
# Running on an RTX3090 it takes around 3 minutes to enhance a tile to 28m resolution,
# including training the interpolation model.
#

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if True:
        # Get lat from arg.
        lat = int(sys.argv[1])
        print("[ LOG ] Processing lat: " + str(lat))
        
        # Download Murray lab data
        murrayLat = lat-4 # Bottom of band, other convention weirdness in Murray data
        with open('../data/zips/dlscript.bat','w') as f:
            f.write("@ECHO OFF\n")
            for lon in range(-180,180,4):
                lon_st = str(lon).zfill(3+int(0.5-0.5*np.sign(lon)))
                lat_st = str(murrayLat).zfill(2+int(0.5-0.5*np.sign(murrayLat)))
                f.write("curl -O http://murray-lab.caltech.edu/CTX/tiles/beta01/E"+ \
                        lon_st+"/Murray-Lab_CTX-Mosaic_beta01_E"+lon_st+"_N"+lat_st+"_data.zip\n")

        # Uncomment this to download and unpack the CTX tiles.
        # I personally prefer to run this separately...
        #if True:
            # Run
            #os.system("../data/zips/dlscript.bat")
            #os.system("mv *.zip ../data/zips/")
            
            # Unzip
            #os.system("unzip '../data/zips/*.zip'")
    
            # Clear data directory
            #os.system("rm ../data/*.tif")

            # Move new tiffs to data directory
            #os.system("mv *.tif ../data/")

            # Clear download directory
            #os.system("rm Murray*")
            #os.system("rm ../data/zips/Murray*")
            
        # Run enhancement
        
        pool = multiprocessing.Pool(2)
        if True:
            try:
                mola = openDem("../Mars_MGS_MOLA_DEM_mosaic_global_463m.tif")
                if os.path.isdir("../models/model_enhance_-4_-96"):
                    # Only load the enhance model once. Speed things up when running in a batch
                    model_enhance = keras.models.load_model("../models/model_enhance_-4_-96")
                    print(model_enhance.summary())
                else:
                    model_enhance = None
                for lon in range(int(sys.argv[2]), int(sys.argv[3]), 4):
                    enhanceMain(pool, mola, lat, lon, 45, model_enhance)
            except RuntimeError as e:
                print("FAILCODE: "+str(lat) + " " + str(lon))
                print(e)
        pool.close()
        pool.terminate()
