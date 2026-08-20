"""One entry point, several subcommands.

Before this there were twenty-one scripts invoked by path, which is fine for a
repo you wrote and hostile to everyone else. The subcommands are the operations
somebody actually performs: run it, point it at a real gateway, check whether
the machine is set up, see the configuration, run the evidence.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


BOLD, DIM, RED, GREEN, YELLOW = "1", "2", "31", "32", "33"
TICK, CROSS, WARN = c("ok", GREEN), c("FAIL", RED), c("warn", YELLOW)


def _run(script: str, *args: str) -> int:
    return subprocess.call([sys.executable, str(ROOT / script), *args])


# --- doctor ------------------------------------------------------------------

def doctor() -> int:
    """Answer 'is this machine set up' before anything wastes ten minutes."""
    from reclaim.config import credentials, load

    print(c("\n  Reclaim doctor\n", BOLD))
    problems, warnings = 0, 0

    def line(label: str, status: str, detail: str = "") -> None:
        print(f"  {status:<16} {label:<34} {c(detail, DIM)}")

    # Python
    v = sys.version_info
    if (v.major, v.minor) >= (3, 11):
        line("python", TICK, f"{v.major}.{v.minor}.{v.micro}")
    else:
        line("python", CROSS, f"{v.major}.{v.minor} — 3.11+ required (tomllib)")
        problems += 1

    # No third-party imports should be needed at all.
    try:
        import reclaim.compliance, reclaim.policy, reclaim.service  # noqa: F401
        line("core modules import", TICK, "no dependencies needed")
    except Exception as exc:
        line("core modules import", CROSS, f"{type(exc).__name__}: {exc}")
        problems += 1

    cfg = load()
    line("config", TICK if cfg.path else WARN,
         str(cfg.path.relative_to(ROOT)) if cfg.path else "none found, using defaults")
    for w in cfg.warnings:
        line("config warning", WARN, w[:60])
        warnings += 1

    # Generated data
    data = ROOT / "data"
    generated = list(data.glob("*.jsonl")) + list(data.glob("events.json"))
    if generated:
        line("datasets", TICK, f"{len(generated)} files present")
    else:
        line("datasets", WARN, "not generated — run `reclaim eval` or data/*.py")
        warnings += 1

    # Credentials
    key_id, secret = credentials()
    if key_id and secret:
        if key_id.startswith("rzp_test_"):
            line("razorpay credentials", TICK, f"{key_id} (test mode)")
        else:
            line("razorpay credentials", WARN,
                 f"{key_id[:12]}... is NOT a test key — writes are refused")
            warnings += 1
    else:
        line("razorpay credentials", WARN,
             "unset — live mode unavailable, everything else works")
        warnings += 1

    # Gateway reachability, only if there is something to reach.
    if key_id and secret:
        from reclaim.gateway import GatewayError, RazorpayTestGateway
        try:
            t = time.perf_counter()
            gw = RazorpayTestGateway()
            gw.list_payments(count=1)
            line("gateway reachable", TICK,
                 f"{gw.name} in {(time.perf_counter()-t)*1000:.0f}ms")
        except GatewayError as exc:
            line("gateway reachable", CROSS, str(exc)[:60])
            problems += 1

    # Anthropic key is optional and its absence is not a failure.
    line("anthropic api key", TICK if os.environ.get("ANTHROPIC_API_KEY") else WARN,
         "set" if os.environ.get("ANTHROPIC_API_KEY")
         else "unset — model tier falls back to the offline stand-in")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        warnings += 1

    # The safety properties, since they are cheap and are the core claim.
    try:
        from reclaim.verify import PROPERTIES, assert_permissive_is_permissive, check
        assert_permissive_is_permissive()
        failed = [p.name for p in PROPERTIES if not check(p).holds]
        if failed:
            line("compliance kernel", CROSS, f"{len(failed)} properties FAILED")
            problems += 1
        else:
            line("compliance kernel", TICK, f"{len(PROPERTIES)} properties hold")
    except Exception as exc:
        line("compliance kernel", CROSS, f"{type(exc).__name__}: {exc}")
        problems += 1

    print()
    if problems:
        print(c(f"  {problems} problem(s), {warnings} warning(s)\n", RED))
        return 1
    print(c(f"  ready — {warnings} warning(s), none blocking\n", GREEN))
    return 0


def show_config() -> int:
    from reclaim.config import load
    cfg = load()
    print(c(f"\n  config  {cfg.path or 'defaults only'}\n", BOLD))
    for key in sorted(cfg.values):
        print(f"    {key:<30}{str(cfg.values[key]):<14}{c(cfg.source(key), DIM)}")
    for w in cfg.warnings:
        print(c(f"\n  warning: {w}", YELLOW))
    print()
    return 0


SUITES = {
    "verify": "eval/run_verification.py", "mutation": "eval/run_mutation.py",
    "adversarial": "eval/run_adversarial.py", "security": "eval/run_security_eval.py",
    "diagnosis": "eval/run_diagnosis_eval.py", "detect": "eval/run_detect_eval.py",
    "calibration": "eval/run_calibration.py", "recovery": "eval/run_eval.py",
    "ceiling": "eval/run_ceiling.py", "causal": "eval/run_causal_eval.py",
    "tuning": "eval/run_tuning.py", "sequencer": "eval/run_sequencer_eval.py",
    "receivables": "eval/run_receivables_eval.py", "carts": "eval/run_carts_eval.py",
    "endtoend": "eval/run_end_to_end.py", "latency": "eval/run_latency.py",
    "llm": "eval/run_llm_eval.py",
}



# Subcommands that forward every remaining argument to a script. argparse's
# REMAINDER looked right for this and is not: it swallows leading options, so
# `reclaim run --limit 3` failed with "unrecognized arguments" while
# `reclaim run 3` worked. Slicing sys.argv is less clever and actually correct.
PASSTHROUGH = {
    "run": "serve.py",
    "live": "eval/run_realtime_test.py",
    "test": "tests/test_all.py",
}

HELP = """usage: reclaim <command> [options]

  doctor              check this machine is set up
  config              show settings and where each came from
  run [options]       run the agent on a stream of events
  live [options]      run against Razorpay test mode
  eval [suite]        run the evidence — all suites, or one by name
  test                run the regression suite

  Any options after a command are passed straight through:

    reclaim run --rate 4 --limit 40 --gateway mock
    reclaim live --execute --limit 3
    reclaim eval ceiling

  suites: {suites}
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP.format(suites=", ".join(sorted(SUITES))))
        return 0

    command, rest = argv[0], argv[1:]

    if command in PASSTHROUGH:
        return _run(PASSTHROUGH[command], *rest)
    if command == "doctor":
        return doctor()
    if command == "config":
        return show_config()
    if command == "eval":
        if not rest:
            return subprocess.call([str(ROOT / "run_all.sh")])
        suite, extra = rest[0], rest[1:]
        if suite not in SUITES:
            print(f"unknown suite {suite!r}\n\navailable:")
            for name in sorted(SUITES):
                print(f"  {name}")
            return 1
        return _run(SUITES[suite], *extra)

    print(f"unknown command {command!r}\n")
    print(HELP.format(suites=", ".join(sorted(SUITES))))
    return 1
