#!/bin/bash

if [ -z "$1" ]; then
    echo "USAGE: $0 PATH-CSV-rank0.csv"
    echo "Expect 8 ranks in total!"
    echo "Stores to [PATH-CSV]-all.csv"
    exit 1
fi

file=$1
target="${file/0.csv/-all.csv}"

cp "$file" "$target"

for i in {1..7}; do
    tail -n +2 "${file/0.csv/$i.csv}" >>"$target"
done
