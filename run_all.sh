#!/usr/bin/env bash
# Everything, in dependency order. No arguments, no dependencies, no network.
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n\033[1m===  %s  ===\033[0m\n\n' "$1"; }

step "1/13  regression tests"
python3 tests/test_all.py

step "2/13  generate data"
python3 data/generate.py
python3 data/events.py
python3 data/panel.py
python3 data/cycles.py
python3 data/receivables.py

step "3/13  verify the compliance kernel"
python3 eval/run_verification.py

step "4/13  mutation-test the verifier"
python3 eval/run_mutation.py

step "5/13  adversarial safety"
python3 eval/run_adversarial.py

step "6/13  security"
python3 eval/run_security_eval.py

step "7/13  detection"
python3 eval/run_detect_eval.py

step "8/13  recovery and causal lift"
python3 eval/run_eval.py
python3 eval/run_causal_eval.py

printf '\n\033[1mAll suites completed.\033[0m\n'
