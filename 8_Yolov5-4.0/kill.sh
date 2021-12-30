#!/bin/bash
ps -ef | grep hgb_yolov5_by
ps -ef | grep hgb_yolov5_by | awk '{print $2}' | xargs kill -9