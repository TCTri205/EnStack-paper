# Báo Cáo Kiểm Tra Tham Số (Parameter Audit Report)

**Ngày kiểm tra**: 17/01/2026  
**Mục tiêu**: Đảm bảo tất cả tham số khớp với bài báo EnStack (2411.16561v1)

---

## 📋 Tham Số Chuẩn Theo Paper

### Base Models Training (Section IV-A)
| Parameter | Value | Source |
|-----------|-------|--------|
| Batch Size | 16 | Table I, Page 4 |
| Epochs | 10 | Table I, Page 4 |
| Learning Rate | 2×10⁻⁵ | Table I, Page 4 |
| Optimizer | AdamW | Section IV-A |
| Max Token Length | 512 | Table I, Page 4 |
| Seed | 42 (standard) | - |

### Meta-Classifiers (Table III)

**Logistic Regression:**
- Max Iterations: 200
- Solver: liblinear

**Random Forest:**
- Number of Estimators: 200
- Max Depth: 10

**SVM:**
- Kernel: RBF
- Probability: True
- Random State: 42

**XGBoost:**
- Number of Estimators: 100
- Learning Rate: 0.1
- Max Depth: 6
- Eval Metric: mlogloss

---

## ✅ Kết Quả Kiểm Tra

### 1. `configs/config.yaml` (Production Config)
**Status**: ✅ PASS - Đã đúng 100%

```yaml
training:
  batch_size: 16      ✅
  epochs: 10          ✅
  learning_rate: 2e-5 ✅
  max_length: 512     ✅
  seed: 42            ✅
```

Meta-classifier params: ✅ Đã cấu hình đầy đủ

---

### 2. `configs/config_local.yaml` (Local Testing)
**Status**: ⚠️ INTENTIONAL DEVIATION (For quick testing only)

```yaml
training:
  batch_size: 2       ⚠️ (For local testing)
  epochs: 1           ⚠️ (For local testing)
  max_length: 64      ⚠️ (For local testing)
```

**Note**: File này được thiết kế để test nhanh trên máy local, KHÔNG dùng cho reproduce paper.

---

### 3. `notebooks/EnStack_Colab_Deployment.ipynb`
**Status**: ⚠️ CẦN SỬA - Cell 7

**Hiện tại**:
```python
EPOCHS = 2           ⚠️ KHÔNG ĐÚNG (should be 10)
BATCH_SIZE = 16      ✅ Đúng
```

**Vấn đề**: Mặc định `EPOCHS = 2` khiến người dùng không reproduce đúng paper.

**Đề xuất**: 
- Mặc định: `EPOCHS = 10` (theo paper)
- Có comment hướng dẫn giảm xuống 2 nếu muốn test nhanh

---

### 4. `notebooks/main_pipeline.ipynb`
**Status**: ✅ PASS

Notebook này load trực tiếp từ `config.yaml`, không có hard-coded values.

---

### 5. `scripts/train.py`
**Status**: ✅ PASS

Script sử dụng argparse để override, nhưng default đọc từ config.yaml.

---

## 🔧 Actions Required

### Priority 1: Fix EnStack_Colab_Deployment.ipynb
```python
# Cell 7 - BEFORE (WRONG):
EPOCHS = 2 # @param {type:"integer"}

# Cell 7 - AFTER (CORRECT):
EPOCHS = 10 # @param {type:"integer"} - Paper default. Use 2-3 for quick testing
```

### Priority 2: Add Warning Comment
Thêm comment rõ ràng trong notebook:
```python
# @markdown ⚠️ **For Paper Reproduction**: Keep EPOCHS=10, BATCH_SIZE=16
# @markdown 📝 **For Quick Testing**: Reduce EPOCHS to 2-3
```

---

## 📊 Summary Table

| File | Batch Size | Epochs | LR | Max Length | Status |
|------|-----------|--------|-----|-----------|--------|
| config.yaml | 16 | 10 | 2e-5 | 512 | ✅ PASS |
| config_local.yaml | 2 | 1 | 2e-5 | 64 | ⚠️ Test Only |
| Colab Deployment | 16 | **2** | - | - | ❌ NEEDS FIX |
| main_pipeline | (uses config) | (uses config) | - | - | ✅ PASS |
| train.py | (uses config) | (uses config) | - | - | ✅ PASS |

---

## ✅ Kết Luận

**Vấn đề duy nhất**: `EnStack_Colab_Deployment.ipynb` có `EPOCHS = 2` thay vì `10`.

**Tác động**: Người dùng chạy notebook này sẽ KHÔNG reproduce đúng kết quả paper (vì chỉ train 2 epochs thay vì 10).

**Mức độ**: CRITICAL - Cần sửa ngay.

---

**Người kiểm tra**: AI Agent  
**Ngày**: 17/01/2026
