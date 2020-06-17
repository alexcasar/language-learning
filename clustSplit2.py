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
doc = pd.read_csv(path+"/"+filename+"/"+filename+"_symMedian.csv")  

varnames = doc.columns

for v in varnames:
    item = list(doc[v])
    items = ' '.join(item)
    corpusdir = path+"/"+filename+"/corpus/"+v+"Median/"
    corpus = corpusdir+v+"Median.txt"
    dictsdir = path+"/"+filename+"/dicts/"
    dicts = dictsdir+v+"Median.vocab"
    parsedir = path+"/"+filename+"/parses/"+v+"Median"
    scoresdir = path+"/"+filename+"/scores/"+v+"Median/"
    scores = scoresdir+v+'Median'

    subprocess.call(['mkdir','-p',corpusdir])
    subprocess.call(['mkdir','-p',dictsdir])
    subprocess.call(['mkdir','-p',parsedir])
    subprocess.call(['mkdir','-p',scoresdir])

    with open(corpus, 'w') as f:
        f.write(items)

    subprocess.call(['./dictionary.sh',corpus,dicts])
    subprocess.call(['./varSplit.sh',dicts,parsedir,corpusdir,scores])




