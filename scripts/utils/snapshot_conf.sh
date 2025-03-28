#!/bin/bash
CONFIG="configs/config.yml"
SHA=$(sha1sum "$CONFIG" | cut -c1-7)
SNAPSHOT_FILE="configs/config_${SHA}.yml"
cp "$CONFIG" "$SNAPSHOT_FILE"
echo "Snapshot saved as: $SNAPSHOT_FILE"