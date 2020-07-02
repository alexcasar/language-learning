# !/usr/bin/env python3

#cd ~/Desktop/snet/gits/lang-learn-repo
#FolderName is the folder for the project, contains the "data" folder inside, with the datafile inside it
#python ./ciscoPL/1_Clusterer.py folderName testName
#python ./ciscoPL/1_Clusterer.py test1 data11

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import sys  
import subprocess
import pywt as pywt
import pathlib

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets

wavMode = pywt.Modes.smooth
wavLevel = 9
clustMin = 7
clustMax = 10
clustIts = 10

#Specifies if testing or if generating outputfile
testing = False
    
path = str(pathlib.Path().absolute())
foldername = sys.argv[1]
filename = sys.argv[2]

datapath = path+"/"+foldername+"/data"

fullpath = datapath+"/"+filename
inpath = fullpath + ".csv"
outpath = fullpath + "Clusters.csv"

doc = pd.read_csv(inpath) 

#Function that gets the signal decomposition (wavelet at each level) of a given variable
def get_signal_decomp(data, w, level):
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(level):
        (a, d) = pywt.dwt(a, w, wavMode)
        ca.append(a)
        cd.append(d)
    
    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    for i, y in enumerate(rec_a):
        out = y

    out = (out-min(out))/(max(out)-min(out))
        
    return out

#Function for printing the wavelets decomposition of a given cluster
def wavPrinter(clusters,clustwav,c):
    test = clusters[clusters['cluster']==c]
    testw = clustwav[clustwav['cluster']==c]
    cols = test.columns
    
    fig = plt.figure(figsize=(20,10))

    for i in range(2,12):
        ax = fig.add_subplot(10, 2, 1 + (i-2) * 2)
        ax.plot(test[cols[int(len(cols)/i)]], 'r')
        ax.set_xlim(0, len(test[cols[int(len(cols)/i)]]) - 1)
        ax.set_ylabel("A%s" % cols[int(len(cols)/i)])
        
        ax = fig.add_subplot(10, 2, 2 + (i-2) * 2)
        ax.plot(testw[cols[int(len(cols)/i)]], 'g')
        ax.set_xlim(0, len(testw[cols[int(len(cols)/i)]]) - 1)
        ax.set_ylabel("D%s" % cols[int(len(cols)/i)]) 

varnames = doc.columns

#To remove the column that corresponds to the anomaly tag
varnames = varnames[:-1]
doc = doc[varnames]

wav = pd.DataFrame(columns=varnames)

#Function that gets the wavelet of each variable
for v in varnames:
    item = list(doc[v])
    wav[v] = get_signal_decomp(item, 'db2', wavLevel)
    
doc = doc.replace(np.nan,0)
wav = wav.replace(np.nan,0)

maxi = 0
maxs = 0
labs = 0

for i in range(clustMin,clustMax):
    for t in range(clustIts):
        kmeans = KMeans(n_clusters=i).fit(wav.T)
        s = metrics.silhouette_score(wav.T, kmeans.labels_, metric='euclidean')
        if s > maxs:
            print(i,s)
            maxi = i
            maxs = s
            labs = kmeans.labels_    
print(labs)

normDoc = doc.copy()
normWav = wav.copy()

for v in varnames:
    normDoc[v] = (normDoc[v]-min(normDoc[v]))/(max(normDoc[v])-min(normDoc[v]))
    
clusters = normDoc.T
clusters['cluster']=labs
clustwav = normWav.T
clustwav['cluster']=labs

#display(clusters.groupby('cluster').count())

clId = clustwav.copy()
clId = clId.iloc[:,-2:]
clId.iloc[:,0] = clId.index
clId = clId.reset_index(drop=True)
clId.columns = ['var','cluster']

if not testing:
    clId.to_csv(outpath,index=False) 
else:
    for r in range(max(labs)):
        wavPrinter(clusters,clustwav,r)
#display(clId)


