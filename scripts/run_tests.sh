#!/bin/bash

for file in identitychain/tests/*.py
do
    python "$file"
done
