# !/usr/bin/env python3
'''Grammar Learner 0.6 tests 2018-09-29: unittest
Run test:
$ cd ~/Desktop/snet/gits/lang-learn-repo/
# python tests/langPCA.py ~/Desktop/snet/gits/lang-learn-repo/alex_tests/data data11 5_5
'''
#from pathlib import Path
#print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import os, sys
import unittest
import pandas as pd
import subprocess
from decimal import Decimal

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path: sys.path.append(module_path)

from src.grammar_learner.utl import UTC
from src.grammar_learner.read_files import check_dir
from src.grammar_learner.learner import learn_grammar
from src.grammar_learner.pqa_table import pqa_meter

path = sys.argv[1]
filename = sys.argv[2]
par = sys.argv[3]
doc = pd.read_csv(path+"/"+filename+"/"+filename+"_PCA_"+par+".csv")  

varnames = doc.columns

for v in varnames:
    print(v)
    input_parses = path+"/"+filename+"/parses/"+v+"_PCA_"+par
    batch_dir = path+"/"+filename+"/lang/"+v+"_PCA_"+par

    subprocess.call(['mkdir','-p',batch_dir])
    try:
        kwargs = {  # defaults
            'input_parses'  :   input_parses,   # path to directory with input parses
            'output_grammar':   batch_dir   ,   # filename or path
            'output_categories' :    ''     ,   # = output_grammar if '' or not set
            'output_statistics' :    ''     ,   # = output_grammar if '' or not set
            'temp_dir'          :    ''     ,   # temporary files = language-learning/tmp if '' or not set
            'parse_mode'    :   'given'     ,   # 'given' (default) / 'explosive' (next)
            'left_wall'     :   '' ,   # '','none' - don't use / 'LEFT-WALL' - replace ###LEFT-WALL###
            'period'        :   False        ,   # use period in links learning: True/False
            'context'       :   2           ,   # 1: connectors / 2,3...: disjuncts
            'window'        :   'mst'       ,   # 'mst' / reserved options for «explosive» parsing
            'weighting'     :   'ppmi'      ,   # 'ppmi' / future options
            'group'         :   True        ,   # group items after link parsing
            'distance'      :   False       ,   # reserved options for «explosive» parsing
            'word_space'    :   'discrete'  ,   # 'vectors' / 'discrete' - no dimensionality reduction
            'dim_max'       :   100         ,   # max vector space dimensionality
            'sv_min'        :   0.1         ,   # minimal singular value (fraction of the max value)
            'dim_reduction' :   'none'      ,   # 'svm' / 'none' (discrete word_space, group)
            'clustering'    :   'group'     ,   # 'kmeans' / 'group'~'identical_entries' / future options
            'cluster_range' :   (2,48,1)    ,   # min, max, step
            'cluster_criteria': 'silhouette',   # optimal clustering criteria
            'cluster_level' :   1.0         ,   # level = 0, 1, 0.-0.99..: 0 - max number of clusters
            'categories_generalization': 'off', # 'off' / 'cosine' - cosine similarity, 'jaccard'
            'categories_merge'      :   0.8 ,   # merge categories with similarity > this 'merge' criteria
            'categories_aggregation':   0.2 ,   # aggregate categories with similarity > this criteria
            'grammar_rules' :   2           ,   # 1: 'connectors' / 2 - 'disjuncts' / 0 - 'words' (TODO?)
            'rules_generalization'  :  'off',   # 'off' / 'cosine' - cosine similarity, 'jaccard'
            'rules_merge'           :   0.8 ,   # merge rules with similarity > this 'merge' criteria
            'rules_aggregation'     :   0.2 ,   # aggregate rules similarity > this criteria
            'tmpath': module_path + '/tmp/',    # legacy, default if not temp_dir
            'verbose': 'max',       		# display intermediate results: 'none', 'min', 'mid', 'max'
            'linkage_limit': 10000,   		# Link Grammar parameter for tests'
            'add_disjunct_costs'    : True,    # add disjunct costs when saving grammar rules to LG dictionary file
    	    'disjunct_cost_function' : 'reverse_count' # 1/disjunct_count
        }
        #print(kwargs)
        response = learn_grammar(**kwargs)
        with open(response['grammar_file'], 'r') as f:
            rules = f.read().splitlines()
        rule_list = [line for line in rules if line[0:1] in ['"', '(']]
    except:
        print("--ERR--",v,"has no links")
