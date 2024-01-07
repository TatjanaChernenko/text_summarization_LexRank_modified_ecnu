#!/bin/sh

for i in $(ls ./../lib/demo_data/*)
do 
    echo $i
    python ./lex_rank_demo.py my-lex-rank --length=3 --file=$i 
done
