#!/bin/sh

for i in $(ls ./lib/test_data/*)
do 
    echo $i
    python ./bin/lex_rank.py my-lex-rank --length=3 --file=$i 
done
