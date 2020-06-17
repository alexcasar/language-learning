# !/usr/bin/env python3

#cd ~/Desktop/snet/gits/lang-learn-repo/alex_tests
#python clustSplitPCA.py ~/Desktop/snet/gits/lang-learn-repo/alex_tests/data testname x_x

import os
import numpy as np
import pandas as pd
import sys  
import subprocess

path = sys.argv[1]
filename = sys.argv[2]
par = sys.argv[3]
doc = pd.read_csv(path+"/"+filename+"/"+filename+"_PCA_"+par+".csv")  

varnames = doc.columns

for v in varnames:
    item = list(doc[v])
    items = ' '.join(item)
    corpusdir = path+"/"+filename+"/corpus/"+v+"_PCA_"+par+"/"
    corpus = corpusdir+v+"_PCA_"+par+".txt"
    dictsdir = path+"/"+filename+"/dicts/"
    dicts = dictsdir+v+"_PCA_"+par+".vocab"
    parsedir = path+"/"+filename+"/parses/"+v+"_PCA_"+par+"/"
    scoresdir = path+"/"+filename+"/scores/"+v+"_PCA_"+par+"/"
    scores = scoresdir+v+'_PCA_'+par

    subprocess.call(['mkdir','-p',corpusdir])
    subprocess.call(['mkdir','-p',dictsdir])
    subprocess.call(['mkdir','-p',parsedir])
    subprocess.call(['mkdir','-p',scoresdir])

    with open(corpus, 'w') as f:
        f.write(items)

    subprocess.call(['./dictionary.sh',corpus,dicts])
    subprocess.call(['./varSplit.sh',dicts,parsedir,corpusdir,scores])




