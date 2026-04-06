#!/bin/bash
# Evaluation script for content recommendation environment
# Runs baseline agent on all three tasks and computes statistics

echo "================================================"
echo "Content Recommendation Baseline Evaluation"
echo "================================================"
echo ""

RESULTS_FILE="baseline_results.log"
> $RESULTS_FILE  # Clear file

for task in easy medium hard; do
    echo "Running task: $task"
    echo "-------------------"
    
    TASK_NAME=$task python inference.py 2>&1 | tee /tmp/${task}_run.log
    
    # Extract [END] line
    END_LINE=$(grep "^\\[END\\]" /tmp/${task}_run.log)
    echo "$END_LINE" >> $RESULTS_FILE
    
    # Parse score
    SCORE=$(echo $END_LINE | sed -E 's/.*score=([0-9.]+).*/\1/')
    echo "Task $task: Score = $SCORE"
    echo ""
done

echo "================================================"
echo "Summary"
echo "================================================"
cat $RESULTS_FILE
echo ""
echo "Results saved to: $RESULTS_FILE"
