#!/bin/bash +x

echo "1111"
echo $1 # blind
echo "2222"
echo $2 # problem.pddl
echo "3333"
echo $3 # domain.pddl

# "planner-scripts/fd-latest-clean -o 'blind' -- problem.pddl domain.pddl"


planner-scripts/limit.sh -t 600 -m 8000000 -- "planner-scripts/fd-latest-clean -o '$1' -- $2 $3"
