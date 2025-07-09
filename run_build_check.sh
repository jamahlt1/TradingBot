#!/bin/bash
# Enhanced build/test runner with logging and dynamic command sourcing
LOG_FILE="cron_test.log"
CONFIG_FILE="build_command.conf"

echo "--- Run started at $(date) ---" >> $LOG_FILE

if [ -f "$CONFIG_FILE" ]; then
    CMD=$(cat $CONFIG_FILE)
    echo "Running command from $CONFIG_FILE: $CMD" >> $LOG_FILE
    eval $CMD >> $LOG_FILE 2>&1
else
    echo "No $CONFIG_FILE found. Please create one with your build/test command." >> $LOG_FILE
    echo "Example: echo 'pytest' > $CONFIG_FILE" >> $LOG_FILE
fi

echo "--- Run ended at $(date) ---" >> $LOG_FILE