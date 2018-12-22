run() {
    echo "$@"
    "$@"
}

run python3 unified.py cupy managed
run python3 unified.py cupy managed warmup
run python3 unified.py cupy
run python3 unified.py cupy warmup
run python3 unified.py
run python3 unified.py warmup
