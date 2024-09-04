#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/usr/src/ultralytics/"
export PYTHONPATH="${PYTHONPATH}:/usr/src/oor_on_edge/"

python3 -u oor_on_edge/performance_monitoring/run_performance_monitoring.py &
python3 -u oor_on_edge/detection_pipeline/run_detection_pipeline.py &
python3 -u oor_on_edge/data_delivery_pipeline/run_data_delivery_pipeline.py &

tail -F /dev/null
