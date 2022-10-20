#!/bin/bash
TESTCASE='test1'
N_CLUSTERS=30

# delete crossval
echo "DELETE CROSSVAL"
python delete_crossval.py -t ${TESTCASE}

# interpolation 
echo "INITIAL INTERPOLATION"
python interpolation.py -t ${TESTCASE}

# interpolation 
echo "FEATURE ENGINEERING"
python feature_engineering.py -t ${TESTCASE}

# clustering 
echo "CLUSTERING"
mkdir /net/so4/landclim/bverena/large_files/climfill_esa/${TESTCASE}/clusters/
python clustering.py -t ${TESTCASE} -c ${N_CLUSTERS}

# regression learning
echo "REGRESSION LEARNING"
for CLUSTER in $(seq 0 ${N_CLUSTERS}); do
    if [[ ! -f /net/so4/landclim/bverena/large_files/climfill_esa/${TESTCASE}/clusters/datacluster_iter_c${CLUSTER}.nc ]]; then 
        echo "file ${YEAR} e${EPOCH}f${FOLD} subitted to queue ..."
        python regression_learning.py -t ${TESTCASE} -c ${CLUSTER}
    else
        echo "file ${year} e${epoch}f${fold} already exists. skipping ..."
    fi
    #exit
done

# postprocessing
