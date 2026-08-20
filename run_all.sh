#!/usr/bin/env bash
# Every suite, in dependency order. No arguments, no dependencies, no network.
set -euo pipefail
cd "$(dirname "$0")"

n=0
total=14
step() { n=$((n+1)); printf '\n\033[1m===  %d/%d  %s  ===\033[0m\n\n' "$n" "$total" "$1"; }

step "generate data"
python3 data/generate.py
python3 data/events.py
python3 data/panel.py
python3 data/cycles.py
python3 data/receivables.py

step "regression tests";             python3 tests/test_all.py
step "verify the compliance kernel"; python3 eval/run_verification.py
step "mutation-test the verifier";   python3 eval/run_mutation.py
step "adversarial safety";           python3 eval/run_adversarial.py
step "security";                     python3 eval/run_security_eval.py
step "model tier and call volume";   python3 eval/run_llm_eval.py
step "detection";                    python3 eval/run_detect_eval.py
step "recovery";                     python3 eval/run_eval.py
step "causal lift";                  python3 eval/run_causal_eval.py
step "mandate retry sequencing";     python3 eval/run_sequencer_eval.py
step "receivables chasing";          python3 eval/run_receivables_eval.py
step "abandoned carts";              python3 eval/run_carts_eval.py
step "end to end";                   python3 eval/run_end_to_end.py

printf '\n\033[1mAll %d suites completed.\033[0m\n' "$total"
