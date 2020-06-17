#!/bin/bash

# Taken from github.com/sbos/AdaGram.jl utils
# Creates a dictionary file with word counts from a plain text file

# Usage: dictionary.sh text_file dictionary_file

tr ' ' '\n' < $1 | sort -S 10G | uniq -c | awk '{print $2" "$1}' > $2
