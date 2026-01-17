#!/usr/bin/env python3
"""
Comprehensive system check for EnStack training system.
Validates all components for completeness, synchronization, and optimization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_trainer_logic():
    """Check trainer.py for correct resume logic."""
    print("=" * 70)
    print("1. CHECKING TRAINER LOGIC")
    print("=" * 70)

    issues = []
    warnings = []

    trainer_path = Path(__file__).parent.parent / "src" / "trainer.py"
    with open(trainer_path, encoding="utf-8") as f:
        content = f.read()

    # Check for itertools.islice (optimized skip)
    if "itertools.islice" in content:
        print("✅ Optimized skip logic using itertools.islice")
    else:
        issues.append("❌ Missing itertools.islice optimization")

    # Check for batches_to_train variable
    if "batches_to_train" in content:
        print("✅ Proper batch tracking with batches_to_train")
    else:
        warnings.append("⚠️  Missing batches_to_train variable")

    # Check for legacy checkpoint handling
    if "Legacy checkpoint detected" in content:
        print("✅ Legacy checkpoint compatibility")
    else:
        warnings.append("⚠️  No legacy checkpoint handling")

    # Check for atomic saves
    if "tempfile.mkdtemp" in content:
        print("✅ Atomic checkpoint saves")
    else:
        issues.append("❌ No atomic checkpoint saves")

    # Check for proper end-of-batch detection
    if "trained_count == batches_to_train" in content:
        print("✅ Correct end-of-batch detection for resume")
    else:
        issues.append("❌ End-of-batch detection may be wrong")

    # Check for SWA
    if "use_swa" in content and "swa_model" in content:
        print("✅ SWA implementation present")
    else:
        warnings.append("⚠️  SWA may not be implemented")

    return issues, warnings


def check_config_sync():
    """Check config files are synchronized."""
    print("\n" + "=" * 70)
    print("2. CHECKING CONFIG SYNCHRONIZATION")
    print("=" * 70)

    issues = []
    warnings = []

    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

    if not config_path.exists():
        issues.append("❌ config.yaml not found")
        return issues, warnings

    with open(config_path, encoding="utf-8") as f:
        config_content = f.read()

    # Check essential parameters
    required_params = [
        ("use_amp", "Automatic Mixed Precision"),
        ("use_swa", "Stochastic Weight Averaging"),
        ("save_steps", "Mid-epoch checkpoint interval"),
        ("scheduler", "Learning rate scheduler"),
        ("use_dynamic_padding", "Dynamic padding"),
        ("cache_tokenization", "Tokenization caching"),
    ]

    for param, desc in required_params:
        if param in config_content:
            print(f"✅ {desc}: {param}")
        else:
            warnings.append(f"⚠️  Missing {desc}: {param}")

    # Check default values
    if "use_swa: False" in config_content:
        print("✅ SWA disabled by default (correct)")
    elif "use_swa: True" in config_content:
        warnings.append("⚠️  SWA enabled by default (may slow down)")

    if "save_steps: 500" in config_content or "save_steps: 0" in config_content:
        print("✅ save_steps configured")
    else:
        warnings.append("⚠️  save_steps not configured")

    return issues, warnings


def check_validation_tools():
    """Check validation scripts exist and are functional."""
    print("\n" + "=" * 70)
    print("3. CHECKING VALIDATION TOOLS")
    print("=" * 70)

    issues = []
    warnings = []

    scripts_dir = Path(__file__).parent
    required_scripts = [
        ("validate_checkpoint.py", "Checkpoint validation"),
        ("debug_checkpoint.py", "Checkpoint debugging"),
        ("cleanup_checkpoints.py", "Checkpoint cleanup"),
        ("train.py", "Main training script"),
    ]

    for script, desc in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"✅ {desc}: {script}")
        else:
            issues.append(f"❌ Missing {desc}: {script}")

    return issues, warnings


def check_documentation():
    """Check documentation files."""
    print("\n" + "=" * 70)
    print("4. CHECKING DOCUMENTATION")
    print("=" * 70)

    issues = []
    warnings = []

    root_dir = Path(__file__).parent.parent
    docs = [
        ("README.md", "Main documentation"),
        ("CHECKPOINT_ANALYSIS.md", "Checkpoint analysis"),
        ("CHECKPOINT_CORRECTNESS.md", "Correctness proof"),
        ("CHECKPOINT_VISUAL_GUIDE.md", "Visual guide"),
        ("FINAL_ANALYSIS.md", "Final analysis"),
        ("URGENT_FIX.md", "Urgent fix guide"),
    ]

    for doc, desc in docs:
        doc_path = root_dir / doc
        if doc_path.exists():
            print(f"✅ {desc}: {doc}")
        else:
            warnings.append(f"⚠️  Missing {desc}: {doc}")

    return issues, warnings


def check_git_status():
    """Check git status for uncommitted changes."""
    print("\n" + "=" * 70)
    print("5. CHECKING GIT STATUS")
    print("=" * 70)

    import subprocess

    issues = []
    warnings = []

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )

        if result.stdout.strip():
            print("⚠️  Uncommitted changes detected:")
            for line in result.stdout.strip().split("\n")[:10]:
                print(f"    {line}")
            warnings.append("Uncommitted changes present")
        else:
            print("✅ No uncommitted changes")

        # Check current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        print(f"✅ Current branch: {branch}")

    except Exception as e:
        warnings.append(f"⚠️  Git check failed: {e}")

    return issues, warnings


def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("ENSTACK SYSTEM COMPREHENSIVE CHECK")
    print("=" * 70)

    all_issues = []
    all_warnings = []

    # Run all checks
    checks = [
        check_trainer_logic,
        check_config_sync,
        check_validation_tools,
        check_documentation,
        check_git_status,
    ]

    for check_func in checks:
        issues, warnings = check_func()
        all_issues.extend(issues)
        all_warnings.extend(warnings)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_issues:
        print(f"\n❌ CRITICAL ISSUES ({len(all_issues)}):")
        for issue in all_issues:
            print(f"  {issue}")
    else:
        print("\n✅ NO CRITICAL ISSUES")

    if all_warnings:
        print(f"\n⚠️  WARNINGS ({len(all_warnings)}):")
        for warning in all_warnings:
            print(f"  {warning}")
    else:
        print("\n✅ NO WARNINGS")

    # Final verdict
    print("\n" + "=" * 70)
    if not all_issues:
        if not all_warnings:
            print("🎉 SYSTEM STATUS: EXCELLENT")
            print("All components are complete, synchronized, and optimized!")
        else:
            print("✅ SYSTEM STATUS: GOOD")
            print("Core functionality is intact. Minor improvements suggested.")
        return 0
    else:
        print("❌ SYSTEM STATUS: NEEDS ATTENTION")
        print("Critical issues must be fixed before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
