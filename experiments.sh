#!/bin/bash

python run_conformal_metalearners.py -t 0.1    -b "DR" "IPW" "X" \
                                     -e "IHDP" -q True -v True \
                                     -c 0.1    -w False;

python run_conformal_metalearners.py -t 0.1 -b "DR" "IPW" "X" \
                                     -s "A" -e "Synthetic" -n 1000 \
                                     -d 10  -q True -v True -x 100 \
                                     -c 0.1 -w True;

python run_conformal_metalearners.py -t 0.1 -b "DR" "IPW" "X" \
                                     -s "B" -e "Synthetic" -n 1000 \
                                     -d 10  -q True -v True -x 100 \
                                     -c 0.1 -w True