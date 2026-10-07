#!/usr/bin/env python3
"""
IndicBridge Standalone Test Runner (Zero-Dependency)
Runs all unit test modules in tests/ using standard Python library.
Can be run via:
  python run_tests.py
"""

import os
import sys
import time
import inspect
import importlib

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_MODULES = [
    "tests.test_tokenizer",
    "tests.test_normalizer",
    "tests.test_index",
    "tests.test_tfidf",
    "tests.test_retrieval",
]

C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_BLUE = "\033[94m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"


def run_all_tests():
    print(f"\n{C_BOLD}{C_BLUE}============================================================{C_RESET}")
    print(f"{C_BOLD}INDICBRIDGE TEST SUITE: Zero-Dependency Pure Python Runner{C_RESET}")
    print(f"{C_BOLD}{C_BLUE}============================================================{C_RESET}\n")

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    start_time = time.time()

    for mod_name in TEST_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"{C_RED}[ERROR] Could not import {mod_name}: {e}{C_RESET}")
            failed_tests += 1
            continue

        test_funcs = [
            (name, func)
            for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]

        print(f"{C_BOLD}Module: {mod_name}{C_RESET} ({len(test_funcs)} tests)")

        for name, func in test_funcs:
            total_tests += 1
            try:
                func()
                passed_tests += 1
                print(f"  {C_GREEN}✓ PASSED{C_RESET}: {name}")
            except Exception as e:
                failed_tests += 1
                print(f"  {C_RED}✗ FAILED{C_RESET}: {name} -> {e}")

        print()

    elapsed = time.time() - start_time
    print(f"{C_BOLD}{C_BLUE}============================================================{C_RESET}")
    if failed_tests == 0:
        print(f"{C_BOLD}{C_GREEN}ALL {passed_tests}/{total_tests} TESTS PASSED SUCCESSFULLY! ({elapsed:.3f}s){C_RESET}")
    else:
        print(f"{C_BOLD}{C_RED}{failed_tests} TESTS FAILED out of {total_tests} total.{C_RESET}")
    print(f"{C_BOLD}{C_BLUE}============================================================{C_RESET}\n")

    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
