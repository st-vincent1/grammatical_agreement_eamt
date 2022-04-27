#!/bin/bash

set -e
DATA=data
for SPLIT in train dev test; do
  PREFIX=${DATA}/en-pl/en-pl.${SPLIT}
  # Set this path to whatever is true to you
  BICLEANER=~/Documents/bicleaner
      
  if [ ! -d $BICLEANER/models/en-pl ]; then
          $BICLEANER/utils/download-pack.sh en pl $BICLEANER/models
  fi  
      
  if [ ! -f bicleaner.en-pl ]; then
          paste ${PREFIX}.preproc.en ${PREFIX}.preproc.pl > ${DATA}/bicleaner.en-pl
  fi

  bicleaner-classify ${DATA}/bicleaner.en-pl ${DATA}/classified.en-pl $BICLEANER/models/en-pl/en-pl.yaml --scol 1 --tcol 2        
  rm ${DATA}/bicleaner.en-pl
  echo "pl done"
done

#awk -v FS='\t' -v OFS='\t' '$3>0.7 {print $1"|"$2}' ${DATA}/classified.en-pl > ${DATA}/filtered.en-pl


# After this:
# - sort the bicleaned data and check if there are any duplicates. If yes, then add another duplicate removal step
# - inspect what the data looks like and write a script to move it to the right places
