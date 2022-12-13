#!/bin/bash
TESTCASE='test5'
N_CLUSTERS=30
COUNT=$(expr $N_CLUSTERS - 1)

# delete crossval TODO update things missing
#echo "DELETE CROSSVAL"
#python delete_crossval.py -t ${TESTCASE}
#
## interpolation 
#echo "INITIAL INTERPOLATION"
#python interpolation.py -t ${TESTCASE}
#
## interpolation 
## since scipy uses all cores by default, run with high niceness
#echo "FEATURE ENGINEERING"
#niceness -n 10 python feature_engineering.py -t ${TESTCASE} 
#
## clustering 
#echo "CLUSTERING"
#mkdir /net/so4/landclim/bverena/large_files/climfill_esa/${TESTCASE}/clusters/
#python clustering.py -t ${TESTCASE} -c ${N_CLUSTERS}

# regression learning
echo "REGRESSION LEARNING"
for CLUSTER in $(seq 0 ${COUNT}); do
    TMP=$(expr $N_CLUSTERS - $CLUSTER)
    #CLUSTER_PREFIX=$(printf "%02d" ${CLUSTER})
    CLUSTER_PREFIX=$(printf "%02d" ${TMP})
    if [[ ! -f /net/so4/landclim/bverena/large_files/climfill_esa/${TESTCASE}/clusters/datacluster_iter_c${CLUSTER_PREFIX}.nc ]]; then 
        echo "cluster ${CLUSTER_PREFIX} subitted to queue ..."
        python regression_learning.py -t ${TESTCASE} -c ${CLUSTER_PREFIX}
    else
        echo "cluster ${CLUSTER_PREFIX} already exists. skipping ..."
    fi
    #exit
done

# postprocessing
#echo "POSTPROCESSING"
#python postprocessing.py -t ${TESTCASE}
