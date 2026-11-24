#!/bin/sh
# Phase 4: the agent on the 20 tasks of agent/bench/tasks.jsonl. Run from the repository root,
# after `source smoke/dgx_env.sh`, with the adapters in their folders:
#
#   nohup sh agent/run_evaluations.sh > runs/phase4.log 2>&1 &
#   tail -f runs/phase4_main.log
#
# Brains (one process each: the adapter is loaded once per process)
#   main         base, thinking and rlm with the default adapter (sft_think, thinking on)
#                -> reports/phase4_eval.json
#   sft_nothink  rlm with the phase-2 adapter trained without thinking
#                -> reports/phase4_eval_rlm_sft_nothink.json
#   grpo_pps3    rlm with the phase-1 adapter alone (trained to reason, never to call tools)
#                -> reports/phase4_eval_rlm_grpo_pps3.json
# Memory (rlm with sft_think; the budget of 6000 tokens is the one of 'main')
#   mem0         no compression                 -> reports/phase4_memory_0.json
#   mem4000      a budget that compresses often -> reports/phase4_memory_4000.json
# JSON vs code
#   code         smolagents CodeAgent, base and rlm (sft_think) -> reports/phase4_codeagent.json
#
# Every task is saved as it ends and --resume skips what is done: if the session ends, open
# another and launch the same command again. With no names it runs the brains; pass names to
# run others:  sh agent/run_evaluations.sh mem0 mem4000 code
#
# TAG names a version of the agent, so a new one never overwrites the old results:
#   TAG=v1 sh agent/run_evaluations.sh   ->  reports/phase4_eval_v1.json, runs/phase4_main_v1.log...
#
# BENCH runs another task file (always with its own TAG). The knowledge router (agent/router.py)
# and the RAG agreement filter are on by default since v3; ARCA_AGENT_ROUTER=0 and
# ARCA_RAG_AGREEMENT_FILTER=0 turn them off (v1 and v1k were measured without both):
#   BENCH=agent/bench/tasks_knowledge.jsonl TAG=v1k ARCA_AGENT_ROUTER=0 sh agent/run_evaluations.sh main
set -u
BENCH=${BENCH:-agent/bench/tasks.jsonl}
TAG=${TAG:+_$TAG}
export PYTHONUNBUFFERED=1
mkdir -p runs reports

# name, adapter, rlm thinking (0/1), memory budget, module, output, brains...
run() {
    name=$1 adapter=$2 thinking=$3 budget=$4 module=$5 out=${6%.json}$TAG.json
    shift 6
    if [ ! -f "$adapter/adapter_model.safetensors" ]; then
        echo "== $name: SKIPPED, no $adapter/adapter_model.safetensors"
        return
    fi
    echo "== $name: adapter=$adapter thinking=$thinking budget=$budget router=${ARCA_AGENT_ROUTER:-1} bench=$BENCH brains=$* -> $out"
    ARCA_AGENT_ADAPTER="$adapter" ARCA_AGENT_RLM_THINKING="$thinking" \
        ARCA_AGENT_TOKEN_BUDGET="$budget" \
        uv run --extra train --extra rag --extra agent python -m "$module" --resume \
        --bench "$BENCH" --brains "$@" --out "$out" > "runs/phase4_$name$TAG.log" 2>&1
}

THINK=tool_use/weights/sft_think
wanted="${*:-main sft_nothink grpo_pps3}"
for name in $wanted; do
    case "$name" in
        main) run main $THINK 1 6000 agent.evaluate reports/phase4_eval.json base thinking rlm ;;
        sft_nothink) run sft_nothink tool_use/weights/sft_nothink 0 6000 agent.evaluate \
            reports/phase4_eval_rlm_sft_nothink.json rlm ;;
        grpo_pps3) run grpo_pps3 rlm/weights/grpo_pps3 1 6000 agent.evaluate \
            reports/phase4_eval_rlm_grpo_pps3.json rlm ;;
        mem0) run mem0 $THINK 1 0 agent.evaluate reports/phase4_memory_0.json rlm ;;
        mem4000) run mem4000 $THINK 1 4000 agent.evaluate reports/phase4_memory_4000.json rlm ;;
        code) run code $THINK 1 6000 agent.code_agent reports/phase4_codeagent.json base rlm ;;
        *) echo "unknown: $name" ;;
    esac
done