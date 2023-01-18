#!/bin/bash
TESTCASE='test8'
N_CLUSTERS=10
COUNT=$(expr $N_CLUSTERS - 1)

for CLUSTER in $(seq 0 ${COUNT}); do
    #CLUSTER=$(expr $N_CLUSTERS - $CLUSTER)
    echo "set ${CLUSTER} calc ..."
    python initial_guess.py -t ${TESTCASE} -s ${CLUSTER}
    python interpolation.py -t ${TESTCASE} -s ${CLUSTER}
    python feature_engineering.py -t ${TESTCASE} -s ${CLUSTER}
    python clustering.py -t ${TESTCASE} -s ${CLUSTER} -c 30
    #exit
done
