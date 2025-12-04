#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script.py> [args...]"
    exit 1
fi

PY_SCRIPT="$1"
shift   # shift so "$@" now contains additional arguments to python

# Start gepetto-viewer (gepetto-gui)
echo "Starting gepetto-gui..."
gepetto-gui --width 1920 --height 1080 &

GEPETTO_PID=$!

# Optional: wait a bit for GUI to initialize
sleep 1

# Alternatively: wait until the CORBA server is alive
# (Pinocchio uses this to connect)
# Uncomment if needed:
# until gepetto-gui-client ping 2>/dev/null; do
#     echo "Waiting for gepetto-gui..."
#     sleep 0.5
# done

echo "Launching Python script: $PY_SCRIPT"
python3 "$PY_SCRIPT" "$@"

# Optional: kill gepetto when the script exits
# kill $GEPETTO_PID