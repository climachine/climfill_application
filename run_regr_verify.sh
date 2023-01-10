#!/bin/bash
TESTCASE='test7'
N_SETS=10
N_CLUSTER=8
COUNT=$(expr $N_SETS - 1)
N_CLUSTER=$(expr $N_CLUSTER - 1)

for SET in $(seq 0 ${COUNT}); do
    for CLUSTER in $(seq 0 ${N_CLUSTER}); do
        #CLUSTER=$(expr $N_CLUSTERS - $CLUSTER)
        #SET=$(expr $N_SETS - $SET)
        CLUSTER_PREFIX=$(printf "%02d" ${CLUSTER})
        if [[ ! -f /net/so4/landclim/bverena/large_files/climfill_esa/${TESTCASE}/verification/set${SET}/clusters/datacluster_iter_c${CLUSTER_PREFIX}.nc ]]; then 
            echo "set ${SET} cluster ${CLUSTER} calc ..."
            python regression_learning.py -t ${TESTCASE} -s ${SET} -c ${CLUSTER}
        else
            echo "set ${SET} cluster ${CLUSTER_PREFIX} already exists. skipping ..."
        fi
        #exit
    done
done
