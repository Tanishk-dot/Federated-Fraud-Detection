#!/bin/bash

# Ensure PYTHONPATH allows absolute imports
export PYTHONPATH="../:$PYTHONPATH"

# Use APFS Virtual Drive residing physically on the External HDD
export RAY_TMPDIR="/Volumes/RayData"
mkdir -p $RAY_TMPDIR

echo "Starting sequential training for all 4 datasets..."
echo "Logs will be written to the results/ directory."

echo "1/4: Running PaySim..."
python3 run_single_dataset.py --dataset paysim --rounds 30 --fraction 1.0 > ../results/sim_paysim.log 2>&1
echo "PaySim finished."

echo "2/4: Running IEEE-CIS..."
python3 run_single_dataset.py --dataset ieee_cis --rounds 30 --fraction 1.0 > ../results/sim_ieee_cis.log 2>&1
echo "IEEE-CIS finished."

echo "3/4: Running European CC 2013..."
python3 run_single_dataset.py --dataset european_cc_2013 --rounds 30 --fraction 1.0 > ../results/sim_european_cc_2013.log 2>&1
echo "European CC finished."

echo "4/4: Running Sparkov..."
python3 run_single_dataset.py --dataset sparkov --rounds 30 --fraction 1.0 > ../results/sim_sparkov.log 2>&1
echo "Sparkov finished."

echo "All simulations completed successfully! Sequential run finished."
