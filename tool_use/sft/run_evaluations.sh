#!/bin/sh
# Phase 2 SFT comparison: every model on the benchmark (31 cases) and on the synthetic test
# (200 trajectories of products never trained on). Run from the repository root, after
# `source smoke/dgx_env.sh`, with the adapters of tool_use/sft/train.py in tool_use/weights/.
#
#   nohup sh tool_use/sft/run_evaluations.sh > /dev/null 2>&1 &
#
# Reports go to reports/phase2_sft_<name>_{bench,synth}.json, logs to runs/.
# Pass names to run only some: sh tool_use/sft/run_evaluations.sh sft_think sft_nothink
set -u
export PYTHONUNBUFFERED=1
mkdir -p runs reports

run() {  # name, adapter ('' for none), thinking (0/1)
    echo "== $1: adapter='$2' thinking=$3"
    ARCA_TOOLS_ADAPTER="$2" ARCA_TOOLS_THINKING="$3" uv run --extra train \
        python -m tool_use.evaluate --bench tool_use/bench/cases.jsonl \
        --out "reports/phase2_sft_$1_bench.json" > "runs/phase2_sft_$1_bench.log" 2>&1
    ARCA_TOOLS_ADAPTER="$2" ARCA_TOOLS_THINKING="$3" uv run --extra train \
        python -m tool_use.sft.evaluate_synthetic --data outputs/tool_sft/trajectories_test.jsonl \
        --out "reports/phase2_sft_$1_synth.json" > "runs/phase2_sft_$1_synth.log" 2>&1
}

wanted="${*:-base phase1_nothink phase1_think sft_think sft_nothink}"
for name in $wanted; do
    case "$name" in
        base) run base "" 0 ;;
        phase1_nothink) run phase1_nothink rlm/weights/grpo_pps3 0 ;;
        phase1_think) run phase1_think rlm/weights/grpo_pps3 1 ;;
        sft_think) run sft_think tool_use/weights/sft_think 1 ;;
        sft_nothink) run sft_nothink tool_use/weights/sft_nothink 0 ;;
        *) echo "unknown: $name" ;;
    esac
done
