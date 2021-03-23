#!/bin/bash

if [ "$#" -ne 0 ]
then
    echo "Wrong format: use ./benchmark.sh"
    exit -1
fi

cd Benchmark/

cd Tiny-YOLOv4/
./darknet detector train dfire.data dfire.cfg yolov4-tiny.conv.29 -dont_show -map
for j in $(seq 1000 1000 30000); do
   rm weights/dfire_$j.weights
done

cd YOLOv4/
./darknet detector train dfire.data dfire.cfg yolov4.conv.137 -dont_show -map
for j in $(seq 1000 1000 30000); do
    rm weights/dfire_$j.weights
done

cd ../Tiny-YOLOv3/
./darknet detector train dfire.data dfire.cfg yolov3-tiny.conv.15 -dont_show -map
for j in $(seq 1000 1000 30000); do
   rm weights/dfire_$j.weights
done

cd ../YOLOv3/
./darknet detector train dfire.data dfire.cfg darknet53.conv.74 -dont_show -map
for j in $(seq 1000 1000 30000); do
    rm weights/dfire_$j.weights
done

cd ../YOLOv2/
./darknet detector train dfire.data dfire.cfg darknet19_448.conv.23 -dont_show -map
for j in $(seq 1000 1000 30000); do
    rm weights/dfire_$j.weights
done

cd ../Tiny-YOLOv2/
./darknet detector train dfire.data dfire.cfg yolov2-tiny.conv.13 -dont_show -map
for j in $(seq 1000 1000 30000); do
    rm weights/dfire_$j.weights
done
