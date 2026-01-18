# 🔍 Checkpoint Verification Tools - Quick Start Guide

## Overview

This directory contains comprehensive checkpoint verification and validation tools to ensure training safety and catch corrupted checkpoints early.

---

## 🚀 Quick Usage

### Before Resuming Training

**Always run verification first:**

```bash
# Basic verification (recommended)
python scripts/verify_checkpoint.py --checkpoint_path /path/to/checkpoint

# Strict mode (treat warnings as errors)
python scripts/verify_checkpoint.py --checkpoint_path /path/to/checkpoint --strict
```

### Verification Tools Comparison

| Tool | Purpose | Use Case | Speed |
|------|---------|----------|-------|
| `verify_checkpoint.py` | **Comprehensive verification** | Pre-flight checks before resuming | ~2 seconds |
| `validate_checkpoint.py` | **State interpretation** | Understand checkpoint metadata | ~1 second |
| `quick_verify_checkpoint()` | **Fast integrity check** | Auto-integrated in training | < 1 second |

---

## 📋 Tool Details

### 1. `verify_checkpoint.py` - Comprehensive Verification

**What it checks:**
- ✅ Directory and file existence
- ✅ File integrity (not empty, not corrupted)
- ✅ Training state consistency
- ✅ Optimizer state validation
- ✅ Metadata sanity checks

**Exit codes:**
- `0` = Verification passed
- `1` = Verification failed

**Example output:**
```
======================================================================
CHECKPOINT VERIFICATION REPORT
======================================================================

🔍 CHECK 1: Directory Existence
✅ Directory exists: /path/to/checkpoint

🔍 CHECK 2: Required Files
✅ training_state.pth: 12.45 MB
✅ config.json: 0.00 MB
✅ model.safetensors: 475.50 MB

🔍 CHECK 3: Training State Integrity
✅ Training state loaded successfully
✅ epoch=2
✅ step=0
✅ total_batches=1270

🔍 CHECK 4: Metadata Consistency
✅ Epoch and step values are valid
✅ End-of-epoch checkpoint (epoch 2 completed)

🔍 CHECK 5: Optimizer State
✅ Optimizer has performed 2532 steps
✅ Optimizer steps consistent with metadata

======================================================================
SUMMARY
======================================================================
✅ Passed: 18
⚠️  Warnings: 0
❌ Errors: 0

✅ VERIFICATION PASSED
Checkpoint is valid and safe to resume training from.
```

---

### 2. `validate_checkpoint.py` - State Interpretation

**What it shows:**
- 📊 Checkpoint metadata (epoch, step, batches)
- 🔍 Interpretation (completed vs incomplete)
- 🔧 Optimizer state
- 📁 Model files
- 📝 Resume behavior prediction

**Usage:**
```bash
python scripts/validate_checkpoint.py --checkpoint_path /path/to/checkpoint
```

**Example output:**
```
======================================================================
CHECKPOINT VALIDATION
======================================================================

📊 CHECKPOINT METADATA:
  Epoch: 2
  Step: 0
  Total Batches: 1270

🔍 INTERPRETATION:
  ✅ This is an END-OF-EPOCH checkpoint
  📝 Meaning: Epoch 2 is COMPLETED
  📦 Model has trained on ALL batches 0-1269
  ➡️  When resuming: Will start epoch 3

======================================================================
SUMMARY
======================================================================
✅ This checkpoint represents a COMPLETE epoch 2
✅ Safe to resume - will start epoch 3
✅ No batches will be skipped or duplicated
```

---

### 3. `quick_verify_checkpoint()` - Integrated Check

**Automatically runs in training pipeline:**
```python
# In src/trainer.py - automatically called when resuming
if resume_from:
    quick_verify_checkpoint(resume_from)  # Fast pre-flight check
    loaded_epoch, loaded_step = self.load_checkpoint(resume_from)
```

**What it checks:**
- ✅ Directory exists
- ✅ Required files present (training_state.pth, config.json)
- ✅ Model weights exist (safetensors or bin)
- ✅ Files not empty
- ✅ Training state loadable

**Benefits:**
- Fails fast if checkpoint invalid
- No manual intervention needed
- Minimal overhead (< 1 second)

---

## 🎯 When to Use Each Tool

### Use `verify_checkpoint.py` when:
- ✅ Resuming training after interruption
- ✅ Switching between different machines
- ✅ Checkpoint was saved to Google Drive (may have sync issues)
- ✅ Training failed with checkpoint-related errors
- ✅ Want comprehensive validation before long training run

### Use `validate_checkpoint.py` when:
- ✅ Want to understand checkpoint state
- ✅ Checking if epoch is complete or mid-epoch
- ✅ Debugging resume behavior
- ✅ Verifying expected batches will be trained

### Use `quick_verify_checkpoint()` when:
- ✅ Already integrated (automatic in training)
- ✅ Need fast checks
- ✅ Basic sanity validation sufficient

---

## 🛡️ Safety Features

### Automatic Integration

All verification improvements are **automatically integrated** into training:

1. **Quick verification** before loading checkpoint
2. **Integrity validation** when loading files
3. **Optimizer consistency check** after loading state
4. **Enhanced scheduler logging** during resume

### No Manual Steps Required

Just resume normally:
```python
trainer.train(
    num_epochs=10,
    resume_from="checkpoints/best_model"  # ✅ All checks run automatically
)
```

---

## 📊 Understanding Checkpoint States

### End-of-Epoch Checkpoint
```
epoch=2, step=0, total_batches=1270
→ Status: COMPLETED
→ Resume: Start epoch 3
→ Batches trained: All 1270 batches of epoch 2
```

### Mid-Epoch Checkpoint
```
epoch=3, step=500, total_batches=1270
→ Status: INCOMPLETE (39.4% done)
→ Resume: Continue epoch 3 from step 500
→ Will skip: Batches 0-499
→ Will train: Batches 500-1269
```

---

## ⚠️ Common Warnings and What They Mean

### 1. "Optimizer steps differ from expected"
```
⚠️  WARNING: Optimizer steps (2532) differ from expected (2540) by 8 steps
```
**Meaning:** Small mismatch due to gradient accumulation  
**Action:** Usually harmless if diff < 20 steps

### 2. "Field 'scaler_state_dict' missing"
```
⚠️  WARNING: Field 'scaler_state_dict' missing
```
**Meaning:** Legacy checkpoint without AMP state  
**Action:** AMP may restart from scratch (minor impact)

### 3. "Learning rate very small after fast-forward"
```
⚠️  WARNING: Learning rate very small after fast-forward!
```
**Meaning:** Scheduler has decayed significantly  
**Action:** Verify num_epochs and warmup settings

---

## 🔧 Advanced Usage

### Strict Mode

Treat all warnings as errors:
```bash
python scripts/verify_checkpoint.py \
    --checkpoint_path /path/to/checkpoint \
    --strict
```

Use when:
- Critical production training
- Ensuring perfect checkpoint state
- Debugging subtle issues

### Automation

Integrate into CI/CD:
```bash
# Exit code 0 = success, 1 = failure
python scripts/verify_checkpoint.py --checkpoint_path $CKPT_PATH
if [ $? -eq 0 ]; then
    echo "Checkpoint valid, starting training"
    python train.py --resume_from $CKPT_PATH
else
    echo "Checkpoint invalid, aborting"
    exit 1
fi
```

---

## 📚 Documentation

- **Full Guide:** `docs/checkpoint_improvements.md`
- **Architecture:** See IMPROVEMENT 1-5 in source code
- **Examples:** See test outputs in this README

---

## 🎓 Best Practices

1. **Always verify before resuming long training runs**
   ```bash
   python scripts/verify_checkpoint.py --checkpoint_path /path/to/checkpoint
   ```

2. **Check both `best_model` and `last_checkpoint`**
   ```bash
   # Verify best model
   python scripts/verify_checkpoint.py --checkpoint_path checkpoints/best_model
   
   # Verify latest checkpoint
   python scripts/verify_checkpoint.py --checkpoint_path checkpoints/last_checkpoint
   ```

3. **Use strict mode for critical checkpoints**
   ```bash
   python scripts/verify_checkpoint.py --checkpoint_path /path/to/checkpoint --strict
   ```

4. **Review warnings before resuming**
   - Small optimizer mismatches are usually fine
   - Missing scaler_state_dict is minor
   - Large mismatches (> 100 steps) need investigation

---

## 🐛 Troubleshooting

### "Checkpoint directory does not exist"
- Check path is correct
- Verify Google Drive is mounted (if using Colab)
- Ensure checkpoint save completed successfully

### "Required file missing"
- Checkpoint may be corrupted
- Save was interrupted
- Use previous checkpoint

### "Failed to load training state"
- File corrupted
- Incompatible PyTorch version
- Try loading with `map_location='cpu'`

---

## 📞 Support

For issues or questions:
1. Check `docs/checkpoint_improvements.md`
2. Review training logs
3. Run verification with `--strict` flag
4. Check AGENTS.md for project guidelines

---

**Last Updated:** 2026-01-18  
**Version:** 1.0.0
