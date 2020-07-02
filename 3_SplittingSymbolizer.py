# !/usr/bin/env python3

#cd ~/Desktop/snet/gits/lang-learn-repo
#FolderName is the folder for the project, contains the "data" folder inside, with the datafile inside it
#python ./ciscoPL/3_SplittingSymbolizer.py folderName testName
#python ./ciscoPL/3_SplittingSymbolizer.py test1 data11

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import sys  
import subprocess
import pywt as pywt
import pathlib

from sklearn import metrics
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from mpl_toolkits import mplot3d

#Specifies if testing or if generating outputfile
testing = False

#Specify if you want to set the number of dimensions, or if you want them to be optimized
optimized = False

if optimized:
    #Specify the % of variance you want explained by the pca
    limit = .95
else:
    #Specify the number of dimensions for pca and blocks (resolution per dimension) for symbilizing
    dims = 5
    blocks = 5
 
path = str(pathlib.Path().absolute())
foldername = sys.argv[1]
filename = sys.argv[2]

path = str(pathlib.Path().absolute())
datapath = path+"/"+foldername+"/data"

fullpath = datapath+"/"+filename
datapath = fullpath + ".csv"
inpath = fullpath + "Clusters.csv"
doc = pd.read_csv(datapath) 
clst = pd.read_csv(inpath)

def pcaSymbolize(df,dim=3,block=3):
    """
    Function that symbolizes the numeric signals in the dataframe given the number of dimensions
    after they have gone through the PCA function. 
    
    We first Rescales data to 0-1 ranges with a min-max scaling
    
    We then create the symbolization based on a density approach, 
    given by the number of points within a given quantile, defined by 'blocks'
    
    All symbols will be a letter R followed by a number.
    Since there are n number of dimensions, and each dimension has d divisions
    There are d**n possible symbols
    
    The following code first defines the position (which block) of the point on each dimension
    and then it computes the symbol using the following formula
    
    sumForAllDimensions(blockSize**currentDimension)
    """
    out = df.copy()
    minO = df.copy()
    maxO = df.copy()
    
    for v in out.columns:
        minO[v] = min(out[v])
        maxO[v] = max(out[v])
        out[v] = (out[v]-minO[v])/(maxO[v]-minO[v])
        
    out['sym']="R"
    out['num']=0

    for d in range(dim):
        out['sym'+str(d)]=0
        for b in range(block):
            out.loc[out.loc[:,out.columns[d]]>out.quantile((b+1)/float(block))[out.columns[d]],'sym'+str(d)]=b+1
    
    for c in range(dim):
        out['num']=out['num']+out['sym'+str(c)]*(block**(dim-c-1))
    out['sym']=out['sym']+out['num'].astype(str)
    
    return out,minO,maxO
            

def pcaSymbolizeTest(df,minO,maxO,out2,dim=3,block=3):
    """Similar to the previous function, but symbolizes based on the space generated on the training process"""
    out = df.copy()
    
    for v in out.columns:
        out[v] = (out[v]-minO[v])/(maxO[v]-minO[v])
        
    out['sym']="R"
    out['num']=0
            
    for d in range(dim):
        out['sym'+str(d)]=0
        for b in range(block):
            out.loc[out.loc[:,out.columns[d]]>out2.quantile((b+1)/float(block))[out2.columns[d]],'sym'+str(d)]=b+1
    
    for c in range(dim):
        out['num']=out['num']+out['sym'+str(c)]*(block**(dim-c-1))
    out['sym']=out['sym']+out['num'].astype(str)
    
    return out
            

def pcaPlot(df,dim=3):
    out = df.copy()
    
    for v in out.columns:
        out[v] = (out[v]-min(out[v]))/(max(out[v])-min(out[v]))
        
    if dim==3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(out[out.columns[0]], out[out.columns[1]], out[out.columns[2]], c=out[out.columns[-1]], cmap='coolwarm');

def pcaPlotTest(df,minO,maxO,dim=3):
    out = df.copy()
    
    for v in out.columns[:-1]:
        out[v] = (out[v]-minO[v])/(maxO[v]-minO[v])
        
    if dim==3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_zlim([0,1])
        ax.scatter3D(out[out.columns[0]], out[out.columns[1]], out[out.columns[2]], c=out[out.columns[-1]], cmap='coolwarm');

dataT = doc.copy()

for v in dataT.columns:
    dataT[v] = (dataT[v]-min(dataT[v]))/(max(dataT[v])-min(dataT[v]))

dataT = dataT.replace(np.nan,0)
dataT1 = dataT.iloc[:int(len(dataT)/2),:]
dataT2 = dataT.iloc[int(len(dataT)/2):,:]

dataTest1 = dataT1.T
dataTest1 = dataTest1.iloc[:-1,:]
dataTest1['clst']=list(clst['cluster'])

dataTest2 = dataT2.T
dataTest2 = dataTest2.iloc[:-1,:]
dataTest2['clst']=list(clst['cluster'])

data = dataTest1

for c in range(max(data['clst'])+1):
    sub = data.loc[data.loc[:,'clst']==c,data.columns[:-1]]
    sub = sub.T

    #Uncomment if you want to see the variance explanation ratio
    #pcaT = PCA().fit(sub)
    #plt.figure()
    #plt.plot(np.cumsum(pcaT.explained_variance_ratio_))
    #varsum = pd.DataFrame(np.cumsum(pcaT.explained_variance_ratio_))
    
    #Get parameters for PCA and symbolization
    if optimized:
        dims = min(10,max(3,len(varsum.loc[limit>varsum.loc[:,0],0])))
        blocks = max(3,min(10,int(50/dims)))

    #----------------------------------TRAINING PCA--------------------------------
    #Perform PCA given the parameters
    pca = PCA(n_components=dims)
    pca.fit(sub)
    newSubs = pca.transform(sub)
    newSubs = pd.DataFrame(newSubs)
    newSub,minO,maxO = pcaSymbolize(newSubs,dims,blocks)

    data = data.T
    data['sym'+str(c)]=0
    data.loc[:len(newSub['sym']),'sym'+str(c)]=list(newSub['sym'])
    data = data.T

    subput = newSubs.copy()
    subput['anomaly']=list(doc.loc[:int(len(doc['anomaly'])/2)-1,'anomaly'])
    pcaPlot(subput)
    #----------------------------------TRAINING PCA--------------------------------
    #----------------------------------TESTING PCA--------------------------------
    sub = dataTest2.loc[dataTest2.loc[:,'clst']==c,dataTest2.columns[:-1]]
    sub = sub.T
    newSubs = pca.transform(sub)
    newSubs = pd.DataFrame(newSubs)
    newSub = pcaSymbolizeTest(newSubs,minO,maxO,newSub,dims,blocks)

    dataTest2 = dataTest2.T
    dataTest2['sym'+str(c)]=0
    dataTest2.loc[:len(newSub['sym']),'sym'+str(c)]=list(newSub['sym'])
    dataTest2 = dataTest2.T

    subput = newSubs.copy()
    subput['anomaly']=list(doc.loc[int(len(doc['anomaly'])/2):,'anomaly'])
    pcaPlotTest(subput,minO,maxO)
    #----------------------------------TESTING PCA--------------------------------

output = data.T
lim = -int(max(data['clst'])+1)
output = output.loc[output.index[:-1],output.columns[lim:]]

fullOutput = output.copy()

output['anomaly']=list(doc.loc[:int(len(doc['anomaly'])/2)-1,'anomaly'])
outSY = output[output['anomaly']==1]
outSN = output[output['anomaly']==0]

for n in range(len(outSY.columns)):
    groupAns = outSY.groupby(by=outSY.columns[n]).count().iloc[:,0].index
    print('anomaly:\t\t',len(groupAns))
    print(groupAns)
    groupNons = outSN.groupby(by=outSN.columns[n]).count().iloc[:,0].index
    print('nonanomaly:\t\t',len(groupNons))
    print(groupNons)
    listG = list(groupAns)+list(groupNons)
    setG = set(listG)
    print('------------------------------------------------------')

 #----------------------------------TESTING PCA--------------------------------
output = dataTest2.T
lim = -int(max(dataTest2['clst'])+1)
output = output.loc[output.index[:-1],output.columns[lim:]]

fullOutput = fullOutput.append(output)

output['anomaly']=list(doc.loc[int(len(doc['anomaly'])/2):,'anomaly'])
outSY = output[output['anomaly']==1]
outSN = output[output['anomaly']==0]

for n in range(len(outSY.columns)):
    groupAns = outSY.groupby(by=outSY.columns[n]).count().iloc[:,0].index
    print('anomaly:\t\t',len(groupAns))
    print(groupAns)
    groupNons = outSN.groupby(by=outSN.columns[n]).count().iloc[:,0].index
    print('nonanomaly:\t\t',len(groupNons))
    print(groupNons)
    listG = list(groupAns)+list(groupNons)
    setG = set(listG)
    print('------------------------------------------------------')
    
#display(fullOutput)

if not testing:
    output.to_csv(fullpath+"Symbol.csv",index=False) 



