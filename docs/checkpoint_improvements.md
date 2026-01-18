# Checkpoint Verification & Safety Improvements

## 📋 Overview

This document describes the comprehensive checkpoint verification and safety improvements implemented in the EnStack training system. These improvements ensure checkpoint integrity, prevent training failures, and catch corrupted checkpoints early.

---

## 🎯 Improvements Implemented

### **IMPROVEMENT 1: Checkpoint Integrity Validation**

**Location:** `src/trainer.py:_validate_checkpoint_integrity()`

**Purpose:** Validates checkpoint file integrity before loading to prevent training failures.

**Features:**
- ✅ Verifies checkpoint directory exists and is valid
- ✅ Checks all required files are present (`training_state.pth`, `config.json`)
- ✅ Validates model weights exist (either `model.safetensors` or `pytorch_model.bin`)
- ✅ Ensures files are not empty (size > 0)
- ✅ Provides detailed error messages if validation fails

**Benefits:**
- Prevents "file not found" errors mid-training
- Catches corrupted checkpoints before GPU memory allocation
- Provides clear error messages for debugging

**Example Output:**
```
🔍 Validating checkpoint integrity...
  ✅ Found model weights: model.safetensors
✅ Checkpoint integrity validated
✅ Verified model weights loaded from /path/to/checkpoint
```

---

### **IMPROVEMENT 2: Optimizer State Consistency Check**

**Location:** `src/trainer.py:_validate_optimizer_consistency()`

**Purpose:** Validates optimizer step count matches checkpoint metadata to detect inconsistencies.

**Features:**
- ✅ Extracts optimizer step count from state dict
- ✅ Calculates expected steps based on epoch/step metadata
- ✅ Compares actual vs expected with configurable tolerance (default: 20 steps)
- ✅ Warns if mismatch detected with detailed diagnostics
- ✅ Distinguishes between harmless (gradient accumulation) and serious mismatches

**Benefits:**
- Detects checkpoint corruption or inconsistency
- Identifies issues with gradient accumulation configuration
- Helps debug scheduler/optimizer synchronization problems

**Example Output (Normal Case):**
```
  Optimizer Steps: 2532
  Expected Steps: 2540
  ✅ Optimizer state consistent (diff=8 steps, within tolerance=20)
```

**Example Output (Warning Case):**
```
============================================================
⚠️  OPTIMIZER STEP MISMATCH DETECTED
  Actual optimizer steps: 5000
  Expected steps: 2540
  Difference: 2460 steps

  Possible causes:
  1. Gradient accumulation (expected behavior)
  2. Scheduler warmup phase
  3. Training was interrupted mid-batch

  ⚠️  Large mismatch (2460 steps) - checkpoint may be inconsistent!
     Consider retraining from an earlier checkpoint.
============================================================
```

---

### **IMPROVEMENT 3: Enhanced Scheduler Fast-Forwarding**

**Location:** `src/trainer.py:train()` (scheduler fast-forward section)

**Purpose:** Provides detailed logging and validation when fast-forwarding scheduler during resume.

**Features:**
- ✅ Logs detailed calculation of steps to skip
- ✅ Shows learning rate before and after fast-forward
- ✅ Validates learning rate is in expected range
- ✅ Warns if LR is too small or too large
- ✅ Confirms scheduler synchronization with training progress

**Benefits:**
- Ensures correct learning rate when resuming
- Catches scheduler configuration errors
- Provides transparency into resume process

**Example Output:**
```
============================================================
⏩ FAST-FORWARDING SCHEDULER
  Total steps to skip: 2540
  Calculation: (2 epochs × 1270 steps/epoch) + 0 mid-epoch steps
  LR before fast-forward: 2.00e-05
  LR after fast-forward: 1.23e-05
  ✅ Learning rate in expected range
============================================================
```

**Example Warning:**
```
  LR after fast-forward: 1.23e-09
  ⚠️  WARNING: Learning rate very small after fast-forward!
     LR=1.23e-09 - scheduler may have decayed too far
```

---

### **IMPROVEMENT 4: Quick Verification Utility**

**Location:** `src/utils.py:quick_verify_checkpoint()`

**Purpose:** Lightweight checkpoint verification for integration into training scripts.

**Features:**
- ✅ Fast pre-flight checks before loading checkpoint
- ✅ Validates directory and file existence
- ✅ Checks for file corruption (empty files)
- ✅ Verifies training state can be loaded
- ✅ Minimal overhead (completes in < 1 second)

**Benefits:**
- Fails fast if checkpoint is invalid
- Prevents wasted GPU initialization time
- Integrated automatically into training pipeline

**Usage in Training Script:**
```python
from src.utils import quick_verify_checkpoint

# Before resuming training
try:
    quick_verify_checkpoint(checkpoint_path)
except (FileNotFoundError, ValueError) as e:
    logger.error(f"Checkpoint verification failed: {e}")
    raise
```

---

### **IMPROVEMENT 5: Comprehensive Verification Tool**

**Location:** `scripts/verify_checkpoint.py`

**Purpose:** Standalone tool for comprehensive checkpoint validation before resuming training.

**Features:**
- ✅ 5 comprehensive validation checks
- ✅ Detailed reporting with pass/warning/error status
- ✅ Strict mode (treats warnings as errors)
- ✅ Exit codes for automation (0=success, 1=failure)
- ✅ Human-readable summary report

**Validation Checks:**
1. **Directory Existence:** Checkpoint directory exists and is valid
2. **Required Files:** All essential files present and non-empty
3. **Training State:** State dict loads successfully and contains required fields
4. **Metadata Consistency:** Epoch/step/batches are logically consistent
5. **Optimizer State:** Optimizer state is valid and consistent with metadata

**Usage:**

```bash
# Basic verification
python scripts/verify_checkpoint.py --checkpoint_path /path/to/checkpoint

# Strict mode (warnings = errors)
python scripts/verify_checkpoint.py --checkpoint_path /path/to/checkpoint --strict
```

**Example Output:**
```
======================================================================
CHECKPOINT VERIFICATION REPORT
Checkpoint: /content/drive/MyDrive/EnStack_Data/checkpoints/codebert/best_model
Strict Mode: False
======================================================================

🔍 CHECK 1: Directory Existence
======================================================================
✅ Directory exists: /content/drive/.../best_model

🔍 CHECK 2: Required Files
======================================================================
✅ training_state.pth: 12.45 MB - Training state (optimizer, metrics, epoch/step)
✅ config.json: 0.00 MB - Model configuration
✅ model.safetensors: 475.50 MB - Model weights (SafeTensors format - preferred)

🔍 CHECK 3: Training State Integrity
======================================================================
✅ Training state loaded successfully
✅ epoch=2 (Current epoch number)
✅ step=0 (Current step within epoch)
✅ 'optimizer_state_dict' present (Optimizer state)
✅ best_val_f1=0.7806 (Best validation F1 score)
✅ best_val_acc=0.7838 (Best validation accuracy)
✅ total_batches=1270 (good!)

🔍 CHECK 4: Metadata Consistency
======================================================================
✅ Epoch and step values are valid (epoch=2, step=0)
✅ End-of-epoch checkpoint (step=0, epoch 2 completed)
✅ Best validation F1: 0.7806
✅ Best validation accuracy: 0.7838

🔍 CHECK 5: Optimizer State
======================================================================
✅ Found 1 parameter group(s)
✅ Optimizer has performed 2532 steps
✅ Optimizer steps (2532) consistent with metadata (expected ~2540, diff=8)

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

## 🔄 Integration into Training Pipeline

The improvements are automatically integrated into the training pipeline:

```python
# In src/trainer.py:train()
if resume_from:
    # IMPROVEMENT 4: Quick verification (automatic)
    quick_verify_checkpoint(resume_from)
    
    # IMPROVEMENT 1: Integrity validation (in load_checkpoint)
    loaded_epoch, loaded_step = self.load_checkpoint(resume_from)
    
    # IMPROVEMENT 2: Optimizer consistency (in load_checkpoint)
    # Automatically validates optimizer state
    
    # IMPROVEMENT 3: Enhanced scheduler logging (in train)
    # Detailed fast-forward logging
```

---

## 📊 Validation Logic Explained

### Checkpoint State Interpretation

**End-of-Epoch Checkpoint:**
```python
epoch=2, step=0, total_batches=1270
→ Epoch 2 COMPLETED (all 1270 batches trained)
→ Resume from START of epoch 3
```

**Mid-Epoch Checkpoint:**
```python
epoch=3, step=500, total_batches=1270
→ Epoch 3 INCOMPLETE (500/1270 batches trained)
→ Resume from step 500 of epoch 3
→ Will skip batches 0-499, train batches 500-1269
```

### Optimizer Step Validation

**Calculation:**
```python
# For end-of-epoch checkpoint (step=0)
expected_opt_steps = epoch × batches_per_epoch

# For mid-epoch checkpoint (step>0)
expected_opt_steps = (epoch-1) × batches_per_epoch + step
```

**Tolerance:**
- Default tolerance: 20 steps
- Adaptive tolerance: max(20, batches_per_epoch // 10)
- Accounts for gradient accumulation and warmup

---

## 🛡️ Error Prevention

### Common Issues Caught

1. **Missing Files:**
   ```
   ❌ ERROR: Required file missing: training_state.pth
   ```

2. **Empty Files:**
   ```
   ❌ ERROR: File is empty: config.json
   ```

3. **Corrupted State:**
   ```
   ❌ ERROR: Failed to load training state: EOFError
   ```

4. **Optimizer Mismatch:**
   ```
   ⚠️  WARNING: Optimizer steps (5000) differ from expected (2540) by 2460 steps
   ```

5. **Invalid Learning Rate:**
   ```
   ⚠️  WARNING: Learning rate very small after fast-forward!
   ```

---

## 💡 Best Practices

### Before Resuming Training

1. **Run comprehensive verification:**
   ```bash
   python scripts/verify_checkpoint.py --checkpoint_path /path/to/checkpoint
   ```

2. **Check output logs for warnings:**
   - Small optimizer mismatches (< 20 steps) are usually harmless
   - Large mismatches (> 100 steps) should be investigated

3. **Verify checkpoint type:**
   - End-of-epoch: `step=0` → Safe to resume
   - Mid-epoch: `step>0` → Will re-train some batches (expected)

### During Training

1. **Monitor scheduler logs:**
   - Verify learning rate is reasonable after resume
   - Check fast-forward calculation matches expectations

2. **Save checkpoints at epoch boundaries:**
   - End-of-epoch checkpoints are safest for resuming
   - Mid-epoch checkpoints useful for crash recovery

3. **Keep multiple checkpoints:**
   - `best_model`: Best validation performance
   - `last_checkpoint`: Latest training state
   - `recovery_checkpoint`: Mid-epoch recovery point

---

## 🔧 Configuration Options

### Optimizer Consistency Tolerance

```python
# In _validate_optimizer_consistency()
tolerance = 20  # Maximum allowed step difference

# To adjust, modify this value or pass as parameter
self._validate_optimizer_consistency(state, expected_steps, tolerance=50)
```

### Strict Verification Mode

```bash
# Treat all warnings as errors
python scripts/verify_checkpoint.py --checkpoint_path /path/to/checkpoint --strict
```

---

## 📈 Performance Impact

- **Quick Verification:** < 1 second overhead
- **Integrity Validation:** Negligible (< 0.5 seconds)
- **Optimizer Consistency:** Negligible (< 0.1 seconds)
- **Enhanced Logging:** Negligible

**Total Resume Overhead:** < 2 seconds (minimal compared to checkpoint loading time)

---

## 🎯 Summary

These improvements provide:

1. ✅ **Early Error Detection:** Catch corrupted checkpoints before training starts
2. ✅ **Detailed Diagnostics:** Clear error messages for debugging
3. ✅ **Consistency Validation:** Ensure optimizer/scheduler state is synchronized
4. ✅ **Transparency:** Detailed logging of resume process
5. ✅ **Safety:** Prevent training failures and wasted compute time

All improvements are **non-breaking** and **automatically integrated** into the existing training pipeline.

---

## 📚 Related Files

- `src/trainer.py`: Main trainer with integrated validation
- `src/utils.py`: Utility functions including `quick_verify_checkpoint()`
- `scripts/verify_checkpoint.py`: Standalone comprehensive verification tool
- `scripts/validate_checkpoint.py`: Original checkpoint state validation tool

---

**Last Updated:** 2026-01-18  
**Version:** 1.0.0
