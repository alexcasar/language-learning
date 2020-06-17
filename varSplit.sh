#!/bin/bash

# Runs stream parser on given corpus, and evaluates resulting parses against given GS using SingNet's parse-evaluator
# at github.com/singnet/language-learning
#

#/home/alexcasar/Desktop/snet/gits/stream-parser-master/src/scripts/alex_tests/varSplit.sh "/home/alexcasar/Desktop/snet/gits/stream-parser-master/src/scripts/alex_tests/data/bgpclear_first/dicts/active-routes-count_leaf1.vocab" "/home/alexcasar/Desktop/snet/gits/stream-parser-master/src/scripts/alex_tests/data/bgpclear_first/parses/active-routes-count_leaf1" "/home/alexcasar/Desktop/snet/gits/stream-parser-master/src/scripts/alex_tests/data/bgpclear_first/corpus/active-routes-count_leaf1" "/home/alexcasar/Desktop/snet/gits/stream-parser-master/src/scripts/alex_tests/data/bgpclear_first/scores/active-routes-count_leaf1/active-routes-count_leaf1.fmi" 3 3


# Evaluate current parses against the gold standard
#parse-evaluator -i -r "$GSPath" -t "$parsesPath"

vocabFilePath=$1;
parsesPath=$2;
corpusPath=$3;
scoresFilePath=$4;

calculateScores=true; # in first pass for each winObserve, store scores
exportScores=true;

# Parse the corpus with current window sizes
java -classpath /home/alexcasar/Desktop/snet/gits/stream-parser-master/src:/home/alexcasar/Desktop/snet/gits/stream-parser-master/src/jar_files/ojalgo-48.0.0.jar mstparser.RunParser 15 15 $calculateScores $exportScores "$vocabFilePath" "$scoresFilePath" "$corpusPath" "$corpusPath" "$parsesPath"

