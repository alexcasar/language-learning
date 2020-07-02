# !/usr/bin/env python3

#cd ~/Desktop/snet/gits/lang-learn-repo
#FolderName is the folder for the project, contains the "data" folder inside, with the datafile inside it
#python ./ciscoPL/6_Predictor.py folderName testName
#python ./ciscoPL/6_Predictor.py test1 data11

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import sys  
import subprocess
import pywt as pywt
import itertools
import pathlib

from sklearn import metrics
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from mpl_toolkits import mplot3d

limit=10
runPredictor=True
anomalyType = 0 #0 es all, 1 es exclusive

path = str(pathlib.Path().absolute())
foldername = sys.argv[1]
filename = sys.argv[2]

fullpath = path+"/"+foldername+"/data"

doc = pd.read_csv(fullpath+"/"+filename+".csv") 

grouped = pd.read_csv(fullpath+"/"+filename+"Symbol.csv") 
grouped['anomaly']=list(doc['anomaly'])
groupedAno = grouped[grouped['anomaly']==1]
groupedAnoN = grouped[grouped['anomaly']==0]

clusters = pd.read_csv(fullpath+"/"+filename+"Clusters.csv")

maxClust = max(clusters['cluster'])
signalClusters=[]

for c in range(maxClust+1): 
    signalClusters=signalClusters + ['sym'+str(c)]

def ruleToPhraseSets(rules):
    words = []
    for r in rules:
        r = r.replace("(","")
        r = r.replace(")","")
        new = r.split(" & ")
        words = words + [new]
    words = list(words for words,_ in itertools.groupby(words))
    
    words2 = []
    for w in words:
        w2 = list(set(w))
        w2.sort()
        words2 = words2 + [w2]
    
    return words2

#- es despues de (o que el otro lo tiene a la izquierda)
#+ es andes de (o que el otro lo tiene al a derecha)
def phrasesFromSets(desc,sets):
    size = len(desc[0])
    phrases = []
    for s in sets:
        if len(s)==1:
            if(s[0][-1]=='-'):
                temp = s[0]
                temp = temp.replace('-','')
                temp = temp.replace(desc[0],'',1)
                phrase = temp + desc[0]
            elif(s[0][-1]=='+'):
                temp = s[0]
                temp = temp.replace('+','')
                temp = temp.replace(desc[0],'',1)
                phrase = desc[0] + temp
            else:
                print("error, no sign")
        else:
            phrase = ""
            unjoined = []
            count = len(s)
            for p in s:
                if(p[-1]=='-'):
                    temp = p[:-1]
                    temp = temp.replace(desc[0],'',1)
                    sub = temp + desc[0]
                elif(p[-1]=='+'):
                    temp = p[:-1]
                    temp = temp.replace(desc[0],'',1)
                    sub = desc[0] + temp
                else:
                    print("error, no sign")
                    
                if(count<len(s)):
                    if (phrase[:size]==sub[-size:]):
                        phrase = sub + phrase[size:]
                    elif (sub[:size]==phrase[-size:]):
                        phrase = phrase[:size] + sub
                    else:
                        if(p[-1]=='-'):
                            phrase = sub + phrase[size:]
                            
                        elif(p[-1]=='+'):
                            phrase = phrase[:size] + sub
                else:
                    phrase = sub
                    
                count = count-1
        phrases = phrases + [phrase]
    return phrases

def getCleanGrammar(s):  
    print(s)
    with open(s, 'r') as f:
        data = f.readlines()
 
    del data[0]
    del data[0]
    del data[0]
    del data[0]
    del data[-1]
    del data[-1]
    del data[-1]
    del data[-1]
    
    full = len(data)
    for r in range(full):
        line = data[full-r-1]
        if(len(line)==1):
            del data[full-r-1]
        elif(line[0]=="%"):
            data[full-r-1] = line[2:-1]
        elif(line[0]=="\""):
            data[full-r-1] = line[1:-3]
        else:
            data[full-r-1] = line[:-2].split(" or ")
    
    clean = []
    mapper = {}
    for r in range(0,len(data),3): 
        joined = [data[r],data[r+1]]
        mapper[joined[1]]=joined[0]
        phraseSets = ruleToPhraseSets(data[r+2])
        phrases = phrasesFromSets(joined,phraseSets)
        joined = joined + [phrases]
        clean = clean + [joined]
        
    return mapper,clean

def getBadRules(clean,maps,keys,exAnomaly,limit,probRisk):
    size = len(clean[0][0])
    maps2 = {}
    for m in maps:
        maps2[maps[m]]=m
    
    
    clean2 = {}
    for c in clean:
        clean2[c[0]]=c[2]
    
    badRules = []
    for rule in clean:
        if rule[1] in exAnomaly:
            badRules = badRules + rule[2]
            
    badRules = list(set(badRules))
    badRules.sort()
    badRuleWord = []
    badRuleCounter = []
    badRuleWordPos = []
    badRuleWordR = []
    badRuleCounterR = []
    badRuleWordPosR = []
       
    r=0
    stop = False
    newRules = badRules
    tester = 1
    while tester != 0:
        badRules = list(set(badRules + newRules))
        badRules.sort()
        newRules = []
        stop = True
        
        tester = len(badRules)
        for rule in badRules:
            if maps2[rule[:size]] in exAnomaly:
                #la primera palabra es anomalia, no hay que agregarle
                tester = tester-1
            elif len(rule)/size >= limit:
                #ya esta muy grande la oracion
                tester = tester-1
            else:
                subRules = []
                for c in clean2[rule[:size]]:
                    if rule[:size]==c[-size:]:
                        subRules = subRules + [c[:-size]+rule]
                
                if(len(subRules)>0):
                    badRules.remove(rule)
                    newRules = newRules + subRules
                else:
                    tester = tester-1
    
    for rule in badRules:
        pos1 = 999
        add1 = ""
        pos2 = 999
        add2 = ""
        risklevel = 0
        for anomaly in exAnomaly:
            loc = rule.find(maps[anomaly])
            if (loc >= 0 and loc < pos1):
                pos1 = loc
                add1 = maps[anomaly]
            prebadword = anomaly
            if (loc >= 0 and probRisk[exAnomaly.index(prebadword)]>risklevel):
                pos2 = loc
                add2 = maps[anomaly]
                risklevel = probRisk[tolerance.index(prebadword)]
        badRuleWord = badRuleWord + [add1]       
        badRuleWordPos = badRuleWordPos + [pos1]
        badRuleCounter = badRuleCounter + [0]
        badRuleWordR = badRuleWordR + [add2]       
        badRuleWordPosR = badRuleWordPosR + [pos2]
        badRuleCounterR = badRuleCounterR + [0]
        r=r+1        
    print(len(badRules))
    return badRules, badRuleWord, badRuleCounter, badRuleWordPos, badRuleWordR, badRuleCounterR, badRuleWordPosR
    

for s in signalClusters:
    try:	
        langFile = fullpath+"/lang/"+s+"/dict.dict"
        maps, clean = getCleanGrammar(langFile)
        maps2 = {}
        for m in maps:
            maps2[maps[m]]=m
        keys = list(maps.keys())
        size = len(grouped[s][0])
        
        anomalies = list(groupedAno.groupby(by=s).count().iloc[:,0].index)
        nonanos = list(groupedAnoN.groupby(by=s).count().iloc[:,0].index)
        exAnomaly = []
        both = []
        for a in anomalies:
            if a not in nonanos:
                exAnomaly = exAnomaly + [a]
            else:
                both = both + [a]
        
        #choose from anomalies=0, exAnomaly=1 
        tols = ['all','exclusive']
        tolId = tols[anomalyType]
        
        probRisk = []
        if(tolId == 'all'):
            tolerance = anomalies
            for t in tolerance:
                ano = list(anomalies).count(t)
                nano = list(nonanos).count(t)
                certainty = 100*(ano/(ano+nano))
                probRisk = probRisk + [certainty]
        else:
            tolerance = exAnomaly
            certainty = 100
            for t in tolerance:
                probRisk = probRisk + [certainty]
            
        badRules, badRuleWord, badRuleCounter, badRuleWordPos, badRuleWordR, badRuleCounterR, badRuleWordPosR = getBadRules(clean,maps,keys,tolerance,limit,probRisk)
        
        rrule = []
        for w in grouped[s]:
            rrule = rrule + [maps[w]]
        fulllist = ''.join(list(grouped[s]))
        fullwordlist = ''.join(rrule)
        
        if(runPredictor):
            a=0
            print(tolerance)
            for word in grouped[s]:
                risk = 0
                a=a+1
                           
                if word in tolerance:
                    certainty = round(probRisk[tolerance.index(word)])
                    if check:
                        badRuleCounter = [-1]*len(badRuleCounter)
                    check = False
                    if (certainty == 100):
                        risk = 2
                        print(a,word,"ERR: Anomalous state",word,'currently active')
                    else:
                        risk = 1
                        print(a,word,"WHI: State",word,"with",certainty,'% chance of being anomaouls, currently active')
                    
                r=0
                minW = 999
                pastword = badRules[0]
                risklevel = 0
                for rule in badRules:
                    if maps[word] in rule:                    
                        if risk == 0:
                            if badRuleCounter[r] == -1 or pastword != word:
                                badRuleCounter[r] = rule.rfind(maps[word])-1
                            elif badRuleCounter[r] < badRuleWordPos[r]:
                                badRuleCounter[r] = badRuleCounter[r] + 1
                            pos = badRuleWordPos[r] - badRuleCounter[r]
                            
                            if badRuleCounter[r] <= badRuleWordPos[r]:
                                minW = pos+1
                                badword = maps2[badRuleWord[r]]
                                badrule = rule
                        elif risk == 1:
                            if badRuleCounterR[r] == -1 or pastword != word:
                                badRuleCounterR[r] = rule.rfind(maps[word])-1
                            elif badRuleCounterR[r] < badRuleWordPosR[r]:
                                badRuleCounterR[r] = badRuleCounterR[r] + 1
                            pos = badRuleWordPosR[r] - badRuleCounterR[r]
                            prebadword = maps2[badRuleWordR[r]]
                            
                            if probRisk[tolerance.index(prebadword)]>risklevel:
                                minW = pos+1
                                badword = prebadword
                                badrule = rule
                                risklevel = probRisk[tolerance.index(prebadword)]
                                            
                    pastword = word
                    r = r+1

                if minW < 999:# and minW > 0:
                    check = True
                    denom = list(grouped[s]).count(word)
                    t = -1
                    k = -1
                    num = 0

                    certainty = round(probRisk[tolerance.index(badword)])
                    softlim = limit
                    for d in range(denom):
                        t = fulllist.find(word, t + 1)
                        k = fulllist.find(badword, t+1, t+softlim*size)
                        if k > -1:
                            num = num+1
                    prob = round(100*num/denom)
                    
                    if prob > 0:
                        if risk == 0:
                            print(a,word,"WLO: State",badword,"with",certainty,'% anomaly risk, in the next',softlim,'words with',prob,'% probability')
                        if risk == 1:
                            fod = " "+" "*len(str(a)) + " "*len(word)
                            print(fod,"WLO: State",badword,"with",certainty,'% anomaly risk, in the next',softlim,'words with',prob,'% probability')
                    elif prob == 0:
                        if risk == 0:
                            print(a,word,"WLO: State",badword,"with",certainty,'% anomaly risk, in the next',softlim,'words with minimal chance')
                        if risk == 1:
                            fod = " "+" "*len(str(a)) + " "*len(word)
                            print(fod,"WLO: State",badword,"with",certainty,'% anomaly risk, in the next',softlim,'words with minimal chance')
                else:
                    if check:
                        badRuleCounter = [-1]*len(badRuleCounter)
                    check = False
                    if word not in tolerance:
                        print(a,word,'Network is healthy')
    except:
        print('No grammar file for',s)    
      


