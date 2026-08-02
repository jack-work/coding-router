#!/usr/bin/env bash
# Live DeepSWE v1.1 arm coverage for the six requested model families.
#
# This is an evaluation-only wave. Each episode runs the official Pier/
# mini-swe-agent loop end to end with one fixed (model, reasoning effort) arm.
# The router policy is not fitted on these outcomes.
#
# Usage on the Azure box:
#   bash live30_deepswe_matrix.sh canary
#   bash live30_deepswe_matrix.sh full
set -Eeuo pipefail

WORK=/nvme/work/deepswe-live
TASKS="$WORK/deep-swe-main/tasks"
PIER="$HOME/.local/bin/pier"
RUN_ID=live30-20260801
MODE=${1:-canary}

case "$MODE" in
  canary)
    OUT="$WORK/jobs-$RUN_ID-canary"
    CONC=2
    FILTER=(-i abs-module-cache-flags -i bandit-structured-nosec-directives)
    ;;
  full)
    OUT="$WORK/jobs-$RUN_ID-full"
    CONC=50
    FILTER=()
    ;;
  *)
    echo "usage: $0 canary|full" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"

ARMS=(
  "luna_low|openai/gpt-5.6-luna|low"
  "luna_medium|openai/gpt-5.6-luna|medium"
  "luna_high|openai/gpt-5.6-luna|high"
  "luna_xhigh|openai/gpt-5.6-luna|xhigh"
  "luna_max|openai/gpt-5.6-luna|max"
  "terra_low|openai/gpt-5.6-terra|low"
  "terra_medium|openai/gpt-5.6-terra|medium"
  "terra_high|openai/gpt-5.6-terra|high"
  "terra_xhigh|openai/gpt-5.6-terra|xhigh"
  "terra_max|openai/gpt-5.6-terra|max"
  "sol_low|openai/gpt-5.6-sol|low"
  "sol_medium|openai/gpt-5.6-sol|medium"
  "sol_high|openai/gpt-5.6-sol|high"
  "sol_xhigh|openai/gpt-5.6-sol|xhigh"
  "sol_max|openai/gpt-5.6-sol|max"
  "opus_low|anthropic/claude-opus-5|low"
  "opus_medium|anthropic/claude-opus-5|medium"
  "opus_high|anthropic/claude-opus-5|high"
  "opus_xhigh|anthropic/claude-opus-5|xhigh"
  "opus_max|anthropic/claude-opus-5|max"
  "fable_low|anthropic/claude-fable-5|low"
  "fable_medium|anthropic/claude-fable-5|medium"
  "fable_high|anthropic/claude-fable-5|high"
  "fable_xhigh|anthropic/claude-fable-5|xhigh"
  "fable_max|anthropic/claude-fable-5|max"
  "sonnet_low|anthropic/claude-sonnet-5|low"
  "sonnet_medium|anthropic/claude-sonnet-5|medium"
  "sonnet_high|anthropic/claude-sonnet-5|high"
  "sonnet_xhigh|anthropic/claude-sonnet-5|xhigh"
  "sonnet_max|anthropic/claude-sonnet-5|max"
)

for spec in "${ARMS[@]}"; do
  IFS='|' read -r name model effort <<< "$spec"
  session="${RUN_ID}-${MODE}-${name}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    continue
  fi
  log="$WORK/${RUN_ID}-${MODE}-${name}.log"
  filter_args="${FILTER[*]}"
  tmux new-session -d -s "$session" \
    "cd '$WORK' && source '$WORK/keys.env' && '$PIER' run -p '$TASKS' $filter_args \
     --agent mini-swe-agent --model '$model' --ak 'reasoning_effort=$effort' \
     --env modal -o '$OUT' --job-name '$name' -n '$CONC' -q -y --max-retries 1 \
     2>&1 | tee '$log'"
  echo "started $name $model@$effort session=$session"
done

echo "mode=$MODE arms=${#ARMS[@]} output=$OUT"
tmux ls | grep "$RUN_ID-$MODE" || true
