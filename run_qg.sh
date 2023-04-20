#!/usr/bin/env bash

g2sdir=g2s_question_generation

# Create required directories
d=`date "+%Y-%m-%d_%H-%M-%S"`
logdir=`pwd`/$g2sdir/logs/$d

mkdir -p $logdir

pushd ~/corenlp
echo "Starting corenlnp server"
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 > $logdir/corenlp.log 2>&1 &
popd

python3 -m $g2sdir.main \
    -task_config $g2sdir/config/teded/qg.yaml \
    -g2s_config $g2sdir/config/teded/new_dependency_ggnn.yaml

echo "Ending corenlp server"
pkill -9 -f edu.stanford.nlp