# !/usr/bin/env python3

#cd ~/Desktop/snet/gits/lang-learn-repo/alex_tests
#python varSplit.py ~/Desktop/snet/gits/lang-learn-repo/alex_tests/data testname

import os
import numpy as np
import pandas as pd
import sys  
import subprocess

path = sys.argv[1]
filename = sys.argv[2]
doc = pd.read_csv(path+"/"+filename+"/"+filename+"_symMean.csv")  

varnames = doc.columns

for v in varnames:
    item = list(doc[v])
    items = ' '.join(item)
    corpusdir = path+"/"+filename+"/corpus/"+v+"Mean/"
    corpus = corpusdir+v+"Mean.txt"
    dictsdir = path+"/"+filename+"/dicts/"
    dicts = dictsdir+v+"Mean.vocab"
    parsedir = path+"/"+filename+"/parses/"+v+"Mean"
    scoresdir = path+"/"+filename+"/scores/"+v+"Mean/"
    scores = scoresdir+v+'Mean'

    subprocess.call(['mkdir','-p',corpusdir])
    subprocess.call(['mkdir','-p',dictsdir])
    subprocess.call(['mkdir','-p',parsedir])
    subprocess.call(['mkdir','-p',scoresdir])

    with open(corpus, 'w') as f:
        f.write(items)

    subprocess.call(['./dictionary.sh',corpus,dicts])
    subprocess.call(['./varSplit.sh',dicts,parsedir,corpusdir,scores])




