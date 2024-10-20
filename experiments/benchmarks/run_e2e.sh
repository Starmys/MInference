# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Load data
wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt

python experiments/benchmarks/benchmark_e2e.py \
    --attn_type minference \
    --context_window 300000
# minference 300000 44.320158338546754
# minference 300000 44.21079638004303