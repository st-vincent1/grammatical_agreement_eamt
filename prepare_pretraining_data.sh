#!/bin/bash

# Download opensubtitles
RAW=data/raw/OpenSubtitles
DATA=data

mkdir -p ${RAW}/en-pl ${RAW}/xml
for lang in en pl; do
  if [ ! -d ${RAW}/xml/${lang} ]; then
    if [ ! -f ${RAW}/xml/${lang}.zip ]; then
      wget "http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/xml/${lang}.zip" -O ${RAW}/xml/${lang}.zip
    fi
    unzip ${RAW}/xml/${lang}.zip -d data/raw && rm ${RAW}/xml/${lang}.zip
  fi
done
if [ ! -f ${RAW}/en-pl/en-pl.xml ]; then
  if [ ! -f ${RAW}/en-pl/en-pl.xml.gz ]; then
    wget "http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/xml/en-pl.xml.gz" -O ${RAW}/en-pl/en-pl.xml.gz
  fi
  zcat ${RAW}/en-pl/en-pl.xml.gz > ${RAW}/en-pl/en-pl.xml
  rm -f ${RAW}/en-pl/en-pl.xml.gz
fi

# Extract bitext from xml files; splits into train, dev, test and saves ids of dev/test
if [ ! -f ${DATA}/en-pl.train.pl ]; then
  python src/prep/extract_bitext.py
fi

# Download Europarl and extract
mkdir -p ${DATA}/europarl_tmp && cd ${DATA}/europarl_tmp
if [ ! -f europarl-v7.pl-en.pl ]; then
  if [ ! -f pl-en.tgz ]; then
    wget https://www.statmt.org/europarl/v7/pl-en.tgz
  fi
  tar -xvf pl-en.tgz && rm pl-en.tgz
fi

# Split Europarl into train, dev, test
TRAIN=575000
LINES=$(cat europarl-v7.pl-en.pl | wc -l)
DEVTEST=$(($LINES - $TRAIN))
DEV=$(($DEVTEST/2))
TEST=$(($DEVTEST - $DEV))
echo $TRAIN training samples, $DEV validation samples, $TEST test samples
for lang in pl en; do
  head -n $TRAIN europarl-v7.pl-en.${lang} > pl-en.train.${lang}
  tail -n $DEVTEST europarl-v7.pl-en.${lang} | head -n $DEV > pl-en.dev.${lang}
  tail -n $DEVTEST europarl-v7.pl-en.${lang} | tail -n $TEST > pl-en.test.${lang}
done

rm europarl-v7.pl-en.*
cd ../..
# Concatenate OpenSubtitles with Europarl

for SET in train dev test; do
  for LANG in en pl; do
    cat europarl_tmp/pl-en.${SET}.${LANG} >> data/en-pl.${SET}.${LANG}
  done
done

rm -r europarl_tmp

for SET in train dev test; do
  python src/prep/preprocess.py -p data/en-pl.${SET}
  paste -d "\t" ${DATA}/en-pl.${SET}.preproc.en ${DATA}/en-pl.${SET}.preproc.pl > ${DATA}/temp
  sort ${DATA}/temp | uniq -u > ${DATA}/temp_no_dups
  awk -F"\t" '{print $1}' ${DATA}/temp_no_dups > ${DATA}/en-pl.${SET}.en
  awk -F"\t" '{print $2}' ${DATA}/temp_no_dups > ${DATA}/en-pl.${SET}.pl
done

mkdir -p ${DATA}/pretrain
# mv this data to raw (since it's still "raw" as test and dev are very big)
mv ${DATA}/en-pl.{dev,test,train}.{en,pl} raw/
rm ${DATA}/*preproc*
rm ${DATA}/temp*

# Copy training data to pretrain
cp ${DATA}/raw/en-pl.train.{en,pl} ${DATA}/pretrain/
# sample dev and test (3k for dev/test)
for SET in dev test; do
  bash src/prep/get_sample_of_bitext.sh data/raw/en-pl.${SET} data/pretrain/en-pl.${SET} 3000
done


# TRAIN SPM MODEL AND ENCODE DATA
python src/prep/encode_and_split.py
