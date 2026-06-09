#!/bin/bash

# GPU Monitoring Script - saves metrics to CSV with timestamps
# Usage: ./monitor_gpus.sh [interval_seconds] [output_file]

INTERVAL=${1:-5}  # Default: sample every 5 seconds
OUTPUT=${2:-logs/gpu_metrics_$(date +%Y%m%d_%H%M%S).csv}
PIDFILE=/tmp/gpu_monitor_$$.pid

echo "Starting GPU monitoring..."
echo "Interval: ${INTERVAL}s"
echo "Output: ${OUTPUT}"
echo "PID file: ${PIDFILE}"

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT")"

# Save PID for cleanup
echo $$ > "$PIDFILE"

# Write CSV header
echo "timestamp,gpu_id,memory_used_mb,memory_total_mb,memory_util_pct,gpu_util_pct,temperature_c,power_draw_w,power_limit_w" > "$OUTPUT"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping GPU monitoring..."
    rm -f "$PIDFILE"
    echo "Metrics saved to: $OUTPUT"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Monitoring loop
while true; do
    TIMESTAMP=$(date +%Y-%m-%d\ %H:%M:%S)
    
    # Query nvidia-smi for all metrics at once
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.memory,utilization.gpu,temperature.gpu,power.draw,power.limit \
        --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r gpu_id mem_used mem_total mem_util gpu_util temp power power_limit; do
        # Trim whitespace
        gpu_id=$(echo "$gpu_id" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        mem_util=$(echo "$mem_util" | xargs)
        gpu_util=$(echo "$gpu_util" | xargs)
        temp=$(echo "$temp" | xargs)
        power=$(echo "$power" | xargs)
        power_limit=$(echo "$power_limit" | xargs)
        
        # Write to CSV
        echo "$TIMESTAMP,$gpu_id,$mem_used,$mem_total,$mem_util,$gpu_util,$temp,$power,$power_limit" >> "$OUTPUT"
    done
    
    # Live display (optional, comment out if too verbose)
    # echo "[$(date +%H:%M:%S)] Logged GPU metrics ($(wc -l < "$OUTPUT") samples)"
    
    sleep "$INTERVAL"
done
