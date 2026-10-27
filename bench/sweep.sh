#!/bin/bash
# Accuracy of the GPU pipeline over the validation pages for each setting given, e.g.
#   bench/sweep.sh "RELV=0 RELEND=0" "RELV=0.15 RELEND=0.1"
# Prints mean CER/WER and CER/WER per set. Needs a release build and PY (python with rapidfuzz).
R=$(cd "$(dirname "$0")/.." && pwd)
B=$R/target/release/ocr-bench
PY=${PY:-python3}
OUT=$R/results/run-sweep
export LC_NUMERIC=C
for cfg in "$@"; do
  line=""; sc=0; sw=0; n=0
  for dir in $R/data/pages/*/; do
    set=$(basename $dir); O=$OUT/$set; rm -rf $O
    (cd $R && env $cfg DUMP=$O MERGE=2 timeout 600 $B $(ls $dir*.png | sort -V) >/dev/null 2>&1)
    rm -f $O/ocrs_*
    r=$(TRUTH=$R/data/truth_${set%%_*}.txt $PY $R/bench/bench.py score $O | tail -1)
    c=$(echo $r | awk '{print $2}'); w=$(echo $r | awk '{print $3}')
    line="$line ${set}:$c/$w"; sc=$(echo "$sc+$c" | bc -l); sw=$(echo "$sw+$w" | bc -l); n=$((n+1))
  done
  printf "%-28s mean CER %5.2f  WER %5.2f |%s\n" "$cfg" $(echo "$sc/$n" | bc -l) $(echo "$sw/$n" | bc -l) "$line"
done
