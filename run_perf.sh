#!/bin/sh
command=$@
perf stat -e cycles,stalled-cycles-frontend,stalled-cycles-backend,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,faults,minor-faults,cs,migrations -r 3 $command
