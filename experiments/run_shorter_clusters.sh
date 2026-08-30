#!/bin/bash

# Ensure PYTHONPATH allows absolute imports
export PYTHONPATH="../:$PYTHONPATH"

# Use system /tmp so Unix Sockets don't exceed length limits
export RAY_TMPDIR="/tmp/ray_fl"
mkdir -p $RAY_TMPDIR

echo "Starting chunked training for all 4 datasets..."
echo "Running shorter clusters to avoid memory and socket length limits."
echo "Logs will be written to the results/ directory."

# Shorter clusters: 3 clients per run, 5 rounds each
NUM_CLIENTS=3
ROUNDS=5
FRACTION=1.0

echo "1/4: Running dataset: paysim (Cluster parameters: $NUM_CLIENTS clients, $ROUNDS rounds)..."
python3 run_single_dataset.py --dataset paysim --rounds $ROUNDS --fraction $FRACTION --num-clients $NUM_CLIENTS > ../results/cluster_paysim.log 2>&1
echo "PaySim finished."

echo "2/4: Running dataset: ieee_cis (Cluster parameters: $NUM_CLIENTS clients, $ROUNDS rounds)..."
python3 run_single_dataset.py --dataset ieee_cis --rounds $ROUNDS --fraction $FRACTION --num-clients $NUM_CLIENTS > ../results/cluster_ieee_cis.log 2>&1
echo "IEEE-CIS finished."

echo "3/4: Running dataset: european_cc_2013 (Cluster parameters: $NUM_CLIENTS clients, $ROUNDS rounds)..."
python3 run_single_dataset.py --dataset european_cc_2013 --rounds $ROUNDS --fraction $FRACTION --num-clients $NUM_CLIENTS > ../results/cluster_europe.log 2>&1
echo "European CC finished."

echo "4/4: Running dataset: sparkov (Cluster parameters: $NUM_CLIENTS clients, $ROUNDS rounds)..."
python3 run_single_dataset.py --dataset sparkov --rounds $ROUNDS --fraction $FRACTION --num-clients $NUM_CLIENTS > ../results/cluster_sparkov.log 2>&1
echo "Sparkov finished."

echo "All clustered simulations completed successfully! Sequential run finished."
