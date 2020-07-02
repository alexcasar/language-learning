#!/bin/bash

# Runs stream parser on given corpus, and evaluates resulting parses against given GS using SingNet's parse-evaluator
# at github.com/singnet/language-learning

vocabFilePath=$1;
parsesPath=$2;
corpusPath=$3;
scoresFilePath=$4;

wsize=15;

calculateScores=true; # in first pass for each winObserve, store scores
exportScores=true;

# Parse the corpus with current window sizes.
java -classpath /home/alexcasar/Desktop/snet/gits/stream-parser-master/src:/home/alexcasar/Desktop/snet/gits/stream-parser-master/src/jar_files/ojalgo-48.0.0.jar mstparser.RunParser $wsize $wsize $calculateScores $exportScores "$vocabFilePath" "$scoresFilePath" "$corpusPath" "$corpusPath" "$parsesPath"

