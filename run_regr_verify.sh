#!/bin/bash
TESTCASE='test8'
N_SETS=10
N_CLUSTER=30
COUNT=$(expr $N_SETS - 1) # pythonic counting
N_CLUSTER=$(expr $N_CLUSTER - 1) # pythonic counting 

for SET in $(seq 0 ${COUNT}); do
    for CLUSTER in $(seq 0 ${N_CLUSTER}); do
        TMP=$(expr $N_CLUSTER - $CLUSTER)
        #SET=$(expr $N_SETS - $SET)
        CLUSTER_PREFIX=$(printf "%02d" ${TMP})
        if [[ ! -f /net/so4/landclim/bverena/large_files/climfill_esa/${TESTCASE}/verification/set${SET}/clusters/datacluster_iter_c${CLUSTER_PREFIX}.nc ]]; then 
            echo "set ${SET} cluster ${CLUSTER_PREFIX} calc ..."
            python regression_learning.py -t ${TESTCASE} -s ${SET} -c ${CLUSTER_PREFIX}
        else
            echo "set ${SET} cluster ${CLUSTER_PREFIX} already exists. skipping ..."
        fi
        #exit
    done
done
