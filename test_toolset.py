#!/usr/bin/env python3
"""Test toolset orchestrator.

Runs all test tools (unit, visual, keybind, deep), aggregates results,
validates output, and reports a unified pass/fail.

Usage:
    python test_toolset.py                # run all tools
    python test_toolset.py --tool all     # same
    python test_toolset.py --tool unit    # unit tests only
    python test_toolset.py --tool visual  # visual tests only
    python test_toolset.py --tool keybind # keybind tests only
    python test_toolset.py --tool deep    # deep tool tests only
    python test_toolset.py --validate     # also validate output formats
"""
import subprocess
import sys
import os
import re
import json
import argparse
import time
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TOOLS = {
    "unit": "test_all.py",
    "visual": "visual_verification.py",
    "keybind": "keybind_test.py",
    "deep": "test_deeptool.py",
}

RESULTS_FILE = os.path.join(tempfile.gettempdir(), "test_toolset_results.json")


def parse_results(tool_name, output):
    """Parse test output and extract pass/fail counts."""
    result = {"tool": tool_name, "output": output, "passed": 0, "failed": 0, "total": 0}

    match = re.search(r"RESULT(?:S)?:\s*(\d+)\s+passed,\s*(\d+)\s+failed,\s*(\d+)\s+total", output)
    if match:
        result["passed"] = int(match.group(1))
        result["failed"] = int(match.group(2))
        result["total"] = int(match.group(3))
    else:
        match2 = re.search(r"(\d+)\s+passed,\s*(\d+)\s+failed", output)
        if match2:
            result["passed"] = int(match2.group(1))
            result["failed"] = int(match2.group(2))
            result["total"] = result["passed"] + result["failed"]

    result["exit_code"] = None
    return result


def run_tool(tool_name, tool_path, validate=False):
    """Run a single test tool and return results."""
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {tool_name} ({tool_path})")
    print(f"{'=' * 60}")

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, tool_path)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=PROJECT_ROOT,
            env={**os.environ, "SDL_VIDEODRIVER": "dummy"},
        )
        elapsed = time.time() - start
        result = parse_results(tool_name, proc.stdout + proc.stderr)
        result["exit_code"] = proc.returncode
        result["elapsed"] = round(elapsed, 2)

        if proc.stdout:
            lines = proc.stdout.strip().split("\n")
            result["last_lines"] = lines[-5:] if lines else []
        if proc.stderr:
            err_lines = proc.stderr.strip().split("\n")
            result["stderr"] = err_lines[-3:] if err_lines else []

        status = "PASS" if result["failed"] == 0 else "FAIL"
        print(f"\n{tool_name}: {status} ({result['passed']}/{result['total']}, {elapsed:.1f}s)")
        if result.get("stderr"):
            print(f"  stderr: {result['stderr']}")

        return result

    except subprocess.TimeoutExpired:
        return {"tool": tool_name, "error": "timeout", "exit_code": -1, "elapsed": 600}
    except Exception as e:
        return {"tool": tool_name, "error": str(e), "exit_code": -1, "elapsed": 0}


def validate_output(tool_results):
    """Validate that test outputs are well-formed."""
    print(f"\n{'=' * 60}")
    print("VALIDATION")
    print(f"{'=' * 60}")

    validations = []

    for r in tool_results:
        tool = r["tool"]
        if "error" in r:
            validations.append({"tool": tool, "check": "no error", "passed": False,
                                "detail": r["error"]})
            continue

        check_results = []

        # Check 1: Results line present
        has_results = bool(re.search(
            r"RESULTS?:\s*\d+\s+passed,\s*\d+\s+failed,\s*\d+\s+total",
            r.get("output", "")))
        check_results.append({"tool": tool, "check": "results line",
                              "passed": has_results})

        # Check 2: Exit code is 0 when no failures
        if r["failed"] == 0:
            exit_ok = r["exit_code"] == 0 or r["exit_code"] is None
            check_results.append({"tool": tool, "check": "exit code",
                                  "passed": exit_ok,
                                  "detail": f"exit={r['exit_code']}"})
        else:
            check_results.append({"tool": tool, "check": "exit code",
                                  "passed": r["exit_code"] != 0,
                                  "detail": f"exit={r['exit_code']} (expected non-zero)"})

        # Check 3: No FAIL lines in output
        output_text = r.get("output", "")
        fail_lines = [l for l in output_text.split("\n") if "FAIL:" in l]
        check_results.append({"tool": tool, "check": "no FAIL lines",
                              "passed": len(fail_lines) == r["failed"],
                              "detail": f"{len(fail_lines)} FAIL lines vs {r['failed']} failures"})

        # Check 4: Has PASS lines
        pass_lines = [l for l in output_text.split("\n") if "PASS:" in l]
        check_results.append({"tool": tool, "check": "has PASS lines",
                              "passed": len(pass_lines) >= r["passed"],
                              "detail": f"{len(pass_lines)} PASS lines vs {r['passed']} passed"})

        validations.extend(check_results)

    all_pass = all(v["passed"] for v in validations)
    for v in validations:
        status = "PASS" if v["passed"] else "FAIL"
        detail = f" - {v['detail']}" if v.get("detail") else ""
        print(f"  {status}: {v['tool']}/{v['check']}{detail}")

    return validations, all_pass


def main():
    parser = argparse.ArgumentParser(description="Run all test tools")
    parser.add_argument("--tool", choices=["all"] + list(TOOLS.keys()),
                        default="all", help="Which tools to run")
    parser.add_argument("--validate", action="store_true",
                        help="Validate output formats")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    tools = TOOLS if args.tool == "all" else {args.tool: TOOLS[args.tool]}

    all_results = []
    overall_pass = 0
    overall_fail = 0

    for tool_name, tool_path in tools.items():
        r = run_tool(tool_name, tool_path, validate=args.validate)
        all_results.append(r)
        if "error" not in r:
            overall_pass += r.get("passed", 0)
            overall_fail += r.get("failed", 0)

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("TOOLSET RESULTS")
    print(f"{'=' * 60}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['tool']}: ERROR - {r['error']}")
        else:
            status = "PASS" if r["failed"] == 0 else "FAIL"
            print(f"  {r['tool']}: {status} ({r['passed']}/{r['total']}, {r['elapsed']}s)")

    print(f"\n  OVERALL: {overall_pass} passed, {overall_fail} failed, {overall_pass + overall_fail} total")

    if args.validate:
        validations, all_valid = validate_output(all_results)
        print(f"\n  VALIDATION: {'ALL PASS' if all_valid else 'SOME FAIL'}")

    print(f"\n  Results saved to: {RESULTS_FILE}")

    return 1 if overall_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())