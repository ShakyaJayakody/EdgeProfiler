#!/bin/bash

PYTHON_PATH="/tinyllama_cpu_env/bin/python"
SCRIPT_PATH="/tinyllama/run_tinyllama_dynamic.py"
OUTPUT_LOG="performance_log.txt"
PERF_OUTPUT="performance_stat.txt"

# Array of prompts and token sizes
PROMPTS=("Hello" "Once upon a time" "The weather today" "A cat sat" "In a galaxy far" "Deep in the forest" "The meaning of life" "Under the ocean" "On a sunny day" "At the library")
TOKENS=(10 20 30 40 50 60 70 80 90 100)

# Clear log files
echo "" > $OUTPUT_LOG
echo "" > $PERF_OUTPUT

# Loop through prompts
for i in {0..9}
do
    prompt="${PROMPTS[$i]}"
    token="${TOKENS[$i]}"
    echo "Running with prompt: $prompt and $token tokens"
    
    sudo perf stat -o temp_perf.txt \
    -e cache-misses,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores,branch-load-misses,branch-loads,dTLB-load-misses,dTLB-loads,dTLB-store-misses,dTLB-stores,iTLB-load-misses,iTLB-loads,node-load-misses,node-loads,node-store-misses,node-stores \
    $PYTHON_PATH $SCRIPT_PATH "$prompt" "$token" > temp_output.txt
    
    echo "Prompt: $prompt" >> $OUTPUT_LOG
    echo "Token Count: $token" >> $OUTPUT_LOG
    echo "Response:" >> $OUTPUT_LOG
    cat temp_output.txt | grep -A 10 "Response:" >> $OUTPUT_LOG
    echo "Performance Stats:" >> $OUTPUT_LOG
    cat temp_perf.txt >> $OUTPUT_LOG
    echo -e "\n-----------------------------\n" >> $OUTPUT_LOG
done

echo "Inference and performance measurement complete. See $OUTPUT_LOG for results."

