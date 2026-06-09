#!/bin/bash

cd datasets

base_url="https://raw.githubusercontent.com/kaustav-basu/IdentifyingCodes/master/Edge-Lists"

files=(
"MadridAdj.txt"
"ParisAdj.txt"
"PhilippineAdj.txt"
"SBG.txt"
"Test.txt"
"ZerkaniAdj.txt"
)

for f in "${files[@]}"; do
    echo "Downloading $f ..."
    curl -L -O "$base_url/$f"
done

echo "Done."
ls -lh