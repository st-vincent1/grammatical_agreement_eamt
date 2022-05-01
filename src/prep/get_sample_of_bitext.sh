#!/bin/bash
# Randomly sample a bitext - extracts N random sentence pairs and writes them to OUT path
IN_PREFIX=$1
OUT_PREFIX=$2
N=$3

paste -d "|" ${IN_PREFIX}.en ${IN_PREFIX}.pl | shuf --random-source=<(yes 42) -n $N > temp
awk -F"|" '{print $1}' temp > ${OUT_PREFIX}.en
awk -F"|" '{print $2}' temp > ${OUT_PREFIX}.pl

rm temp
