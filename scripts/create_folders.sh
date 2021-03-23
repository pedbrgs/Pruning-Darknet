#!/bin/bash

if [ "$#" -ne 1 ]
then
    echo "Wrong format: use ./preparing_directories.sh <pruning_technique>"
    exit -1
fi

cd Models/
mkdir -p $1/
cd $1/

# Iterating over pruning rates
for i in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95; do

    mkdir -p $i/
    chmod 777 $i/
done
