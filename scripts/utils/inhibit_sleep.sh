#!/bin/bash

# Check if a process ID was provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <process_id>"
  exit 1
fi

TARGET_PID="$1"

# Cleanup function to be called on script interrupt
cleanup() {
  echo "Interrupt received, cleaning up..."
  if kill -0 "$INHIBITOR_PID" 2>/dev/null; then
    kill "$INHIBITOR_PID" 2>/dev/null
    wait "$INHIBITOR_PID" 2>/dev/null
    echo "systemd-inhibit process (PID: $INHIBITOR_PID) has been terminated."
  fi
  exit 0
}

# Trap Ctrl+C (SIGINT) to run cleanup
trap cleanup SIGINT

# Start the systemd-inhibit process in the background to prevent sleep.
systemd-inhibit sleep infinity &
INHIBITOR_PID=$!

echo "Launched systemd-inhibit (PID: $INHIBITOR_PID) to prevent sleep."

# Wait until the target process is no longer running.
while kill -0 "$TARGET_PID" 2>/dev/null; do
  sleep 60  # Wait for 1 minute between checks.
done

echo "Target process $TARGET_PID has finished."

# Clean up the background inhibitor process if it's still running.
if kill -0 "$INHIBITOR_PID" 2>/dev/null; then
  kill "$INHIBITOR_PID" 2>/dev/null
  wait "$INHIBITOR_PID" 2>/dev/null
fi

echo "systemd-inhibit process (PID: $INHIBITOR_PID) has been terminated."
