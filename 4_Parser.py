# !/usr/bin/env python3

#cd ~/Desktop/snet/gits/lang-learn-repo
#FolderName is the folder for the project, contains the "data" folder inside, with the datafile inside it
#python ./ciscoPL/4_Parser.py folderName testName
#python ./ciscoPL/4_Parser.py test1 data11

import os
import numpy as np
import pandas as pd
import sys  
import subprocess
import pathlib

foldername = sys.argv[1]
filename = sys.argv[2]

#Script must be within a folder inside the language learning repo
path = str(pathlib.Path().absolute())
fullpath = path+"/"+foldername+"/data"

doc = pd.read_csv(fullpath+"/"+filename+"Symbol.csv")  

varnames = doc.columns

for v in varnames:
    item = list(doc[v])
    items = ' '.join(item)
    corpusdir = fullpath+"/corpus/"+v+"/"
    corpus = corpusdir+v+".txt"
    dictsdir = fullpath+"/dicts/"
    dicts = dictsdir+v+".vocab"
    parsedir = fullpath+"/parses/"+v+"/"
    scoresdir = fullpath+"/scores/"+v+"/"
    scores = scoresdir+v

    subprocess.call(['mkdir','-p',corpusdir])
    subprocess.call(['mkdir','-p',dictsdir])	
    subprocess.call(['mkdir','-p',parsedir])
    subprocess.call(['mkdir','-p',scoresdir])

    with open(corpus, 'w') as f:
        f.write(items)

    subprocess.call(['./ciscoPL/dictionary.sh',corpus,dicts])
    subprocess.call(['./ciscoPL/varSplit.sh',dicts,parsedir,corpusdir,scores])




