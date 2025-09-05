#!/bin/bash
# This script sets up the environment and runs the ensemble attack example.

source .venv/bin/activate

echo "Active Environment:"
which python

echo Experiments Launched

python -m examples.ensemble_attack_example.run_attack

echo Experiments Completed
