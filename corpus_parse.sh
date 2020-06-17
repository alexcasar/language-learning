#!/bin/bash

vocabFilePath=$1;
corpusPath=$2;
parsePath=$3;
scoresFilePath=$4;

java -classpath .::/home/alexcasar/Desktop/snet/gits/stream-parser-master/src:/home/alexcasar/Desktop/snet/gits/stream-parser-master/src/jar_files/ojalgo-48.0.0.jar mstparser.RunParser 3 3 true true "$vocabFilePath" "$scoresFilePath" "$corpusPath" "$corpusPath" "$parsesPath"


