# EnStack: Verification Report & Fix Proposals

This document outlines the findings from the code audit and paper verification (`2411.16561v1.pdf`), along with proposed solutions for the remaining issues.

## 1. Resolved Issues (Fixed)

The following high-priority technical issues have been fixed in the codebase:

1.  **Double Forward Pass Efficiency:**
    *   **Issue:** `src/models.py` was executing the model forward pass twice when calculating loss without label smoothing, causing a 30-50% performance penalty.
    *   **Fix:** Updated `EnStackModel.forward` to calculate loss manually using cached logits, ensuring only one forward pass per step.

2.  **Hard-coded Architecture:**
    *   **Issue:** `src/models.py` and `src/trainer.py` assumed all models had a `.roberta` attribute, breaking compatibility with other architectures (e.g., BERT, DistilBERT).
    *   **Fix:** Implemented dynamic architecture detection and replaced `RobertaForSequenceClassification` with `AutoModelForSequenceClassification`.

---

## 2. Verified Pending Issues

### A. Data Leakage in Stacking Pipeline
*   **Status:** **CRITICAL**
*   **Code Evidence:** `notebooks/EnStack_Colab_Deployment.ipynb` (and `scripts/train.py`) trains base models on the full training set, then uses those *same* models to predict/extract features on the *same* training set for the meta-classifier.
*   **Paper Verification:** Algorithm 1 (page 5) in the paper explicitly describes a **Hold-out (Blending)** strategy:
    > "Split D into D_train, D_val, and D_test... Fine-tune Mk on D_train... For each sample in **D_val**, compute meta-feature vector... Train meta-classifier on D_val."
*   **Conclusion:** The current implementation incorrectly implements stacking, leading to severe overfitting of the meta-classifier (it learns to trust base models too much because they have seen the answers).

### B. GraphCodeBERT Implementation Gap
*   **Status:** **HIGH**
*   **Code Evidence:** `src/dataset.py` treats all input as plain text sequence. `src/models.py` passes only `input_ids` and `attention_mask`.
*   **Paper Verification:** The paper claims GraphCodeBERT "emphasizes structural relationships through data flow graphs".
*   **Technical Reality:** To utilize GraphCodeBERT's structural capabilities, one must parse the code (using Tree-sitter), extract the Data Flow Graph (DFG), and construct special input embeddings. The current implementation treats GraphCodeBERT identical to CodeBERT, negating its structural advantages.

---

## 3. Proposed Solutions

### Proposal 1: Fix Data Leakage (Recommended)

**Option A: Strict Paper Reproduction (Blending)**
*   **Method:** Modify the pipeline to split the Training Data into `Train_Base` (e.g., 80%) and `Train_Meta` (e.g., 20%).
*   **Workflow:**
    1. Train Base Models on `Train_Base`.
    2. Generate predictions (meta-features) on `Train_Meta`.
    3. Train Meta-classifier on `Train_Meta` predictions.
*   **Pros:** Fast, exactly matches the paper's Algorithm 1.
*   **Cons:** Uses less data for training base models.

**Option B: K-Fold Out-of-Fold (OOF) Stacking**
*   **Method:** Use the `fit_with_oof` logic (already in `src/stacking.py` but unused).
*   **Workflow:** Split Train Data into K folds. Train K versions of each base model. Predict on the held-out fold. Combine predictions to form the meta-training set.
*   **Pros:** Utilizes 100% of data, generally yields better performance.
*   **Cons:** Computationally expensive (5x training time), high risk of Colab timeout.

**Recommendation:** Implement **Option A** as the default for `scripts/train.py` to match the paper and ensure stability on Colab. Add an argument `--strategy oof` for advanced users.

### Proposal 2: Enable True GraphCodeBERT Support

**Implementation Plan:**
1.  **Dependency:** Add `tree-sitter` and language grammars (C/C++) to `requirements.txt`.
2.  **Preprocessing:** Create a `DFGParser` class in `src/utils.py` to extract variable relationships.
3.  **Dataset:** Update `VulnerabilityDataset` to accept a mode flag. If `model_type="graphcodebert"`, run DFG parsing and append structural inputs.
4.  **Model:** Update `EnStackModel` to accept `position_ids` (used by GraphCodeBERT to encode graph structure) and forward them to the base model.

**Effort:** High. Recommended as a separate feature branch.

### Proposal 3: Optimize Stacking Workflow

Currently, `scripts/train.py` re-initializes dataloaders multiple times.
*   **Fix:** Cache tokenized datasets to disk (`.cache/`) to allow reuse across different base model training runs, significantly reducing startup time.
