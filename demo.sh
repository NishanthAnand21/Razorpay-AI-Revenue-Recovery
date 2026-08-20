#!/usr/bin/env bash
#
# The pitch demo, paced for narration.
#
#   ./demo.sh              press Enter between beats — you control the pacing
#   ./demo.sh --auto       runs on a timer, no keypresses
#   ./demo.sh --rehearse   fast, no pauses, to check nothing is broken
#
# Everything is pre-warmed before the first title card, so no command hangs on
# camera. One terminal window start to finish: no app switching, no file tree,
# nothing to tour.
set -uo pipefail
cd "$(dirname "$0")"

MODE="manual"
case "${1:-}" in
  --auto) MODE="auto" ;;
  --rehearse) MODE="rehearse" ;;
esac

BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
CYAN=$'\033[36m'; YELLOW=$'\033[33m'; GREEN=$'\033[32m'

beat() {   # beat <seconds-if-auto>
  case "$MODE" in
    manual)   printf "\n${DIM}   ⏎${RESET}"; read -r ;;
    auto)     sleep "${1:-4}" ;;
    rehearse) : ;;
  esac
}

# `clear` needs TERM; fall back to blank lines so the demo still runs when it is
# unset (piped output, CI, a stripped-down shell).
wipe() { command clear 2>/dev/null || printf '\n%.0s' {1..40}; }

card() {   # card <number> <title>
  wipe
  printf "\n\n"
  printf "   ${DIM}%s${RESET}\n" "$1"
  printf "   ${BOLD}%s${RESET}\n\n" "$2"
  [ -n "${3:-}" ] && printf "   ${DIM}%s${RESET}\n" "$3"
  printf "\n"
}

run() {    # run <command...>
  printf "   ${CYAN}\$ %s${RESET}\n\n" "$*"
  "$@" 2>&1 | sed 's/^/   /'
}

# --- pre-warm -----------------------------------------------------------------
wipe
printf "\n   ${DIM}warming up — generating datasets so nothing hangs on camera...${RESET}\n"
python3 data/generate.py >/dev/null 2>&1
python3 data/events.py    >/dev/null 2>&1
python3 data/panel.py     >/dev/null 2>&1
python3 data/cycles.py    >/dev/null 2>&1
python3 data/receivables.py >/dev/null 2>&1
printf "   ${GREEN}ready${RESET}\n"
beat 1

# --- 1. the number that shouldn't be true -------------------------------------
card "01" "\"we recover 31% of failed payments\"" \
     "Every recovery product reports a number like this. Mine does too."
beat 26

card "01" "So I measured what it actually caused." \
     "Regression discontinuity at the NPCI peak-window boundaries."
run python3 eval/run_causal_eval.py 2>&1 | head -25
beat 42

# --- 2. the inversion ---------------------------------------------------------
card "02" "The guardrail was in the wrong place." ""
cat <<'DIAG'
   v1   diagnose ──▶ propose ──▶ veto?      safety depends on the diagnosis
   v2   observe  ──▶ legal set ──▶ model picks inside it

   Every hard constraint — Visa Category 1, NPCI cycle caps, RBI's
   pre-debit notice, the collections window — is decidable from a
   response code, a counter and a clock. None of them needs a model.
DIAG
beat 42

card "02" "So: can a lying model break it?" \
     "A diagnoser that claims 'transient issuer' at confidence 1.0 on every payment."
run python3 eval/run_adversarial.py 2>&1 | head -13
beat 20

# --- 3. live ------------------------------------------------------------------
if [ -n "${RAZORPAY_KEY_ID:-}" ]; then
  card "03" "Live, against Razorpay test mode." \
       "Real objects, real HTTP, real reminders."
  run ./bin/reclaim live --limit 3 --execute
else
  card "03" "Live, over real HTTP." \
       "RAZORPAY_KEY_ID unset — using the local gateway. Same client, same code path."
  run ./bin/reclaim run --gateway mock --limit 30 2>&1 | tail -24
fi
beat 55

# --- 4. the ceiling -----------------------------------------------------------
card "04" "66.8%. Why not 100?" \
     "An oracle that knows the true cause of every payment. Nothing real can beat it."
run python3 eval/run_ceiling.py 2>&1 | sed -n '1,10p;20,40p'
beat 52

# --- 5. the bug ---------------------------------------------------------------
card "05" "The worst bug had nothing to do with the model." ""
cat <<'BUG'
   A gateway timeout does not mean the payment failed.
   It means nobody told us.

   v1 classified timeouts as transient and retried immediately.
   35 double-charge exposures out of 240 payments.

   The diagnosis was right every single time.
   The assumption underneath it was wrong.
BUG
beat 32

card "" "github.com/NishanthAnand21/Razorpay-AI-Revenue-Recovery" \
     "Clone it, run one command. Every number reproduces in two minutes. No dependencies."
beat 14
printf "\n"
