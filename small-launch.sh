#!/bin/bash


task="sokoban"
type="sokoban_image-20000-global-global-2-train"
width_height=""
nb_examples="20000"
conf_folder="05-09T16:42:26.372"
after_sample="sokoban_sokoban_image-20000-global-global-2-train_20000_CubeSpaceAE_AMA4Conv_kltune2"

ouut=$(pypy3.8-v7.3.11-linux64/bin/pypy downward/fast-downward.py --alias lama-first --plan-file $1 $2 $3 --translate-options --invariant-generation-max-candidates 0)

if [[ $ouut == *"] Solution found!"* ]]; then
    counter_solSeven=$((counter_solSeven+1))
fi

echo $ouut


#./ama3plannerNOmodule.py $1 $2 blind 1
#./train_kltune.py dump $task $type $width_height $nb_examples CubeSpaceAE_AMA4Conv kltune2 --hash $conf_folder
# pwdd=$(pwd)
# domain_dir=samples/$after_sample/logs/$conf_folder
# ./pddl-ama3.sh $pwdd/$domain_dir
# then, once you have the domain.pddl, need the problem !
# so, 