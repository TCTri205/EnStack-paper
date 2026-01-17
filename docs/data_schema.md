# Quy định Cấu trúc Dữ liệu (Data Schema)

Tài liệu này mô tả chi tiết định dạng dữ liệu ở các giai đoạn khác nhau của pipeline, 
đảm bảo sự tương thích giữa các module.

> **Xem thêm:**
> - [Methodology](methodology.md) - Phương pháp xử lý dữ liệu trong EnStack
> - [Technical Specification](technical_specification.md) - Class design cho data loading
> - [Deployment Guide](deployment_guide.md) - Hướng dẫn chuẩn bị và upload dữ liệu

---

## 1. Dữ liệu Thô (Raw Data)
*   **Nguồn:** Draper VDISC Dataset.
*   **Định dạng:** HDF5 hoặc CSV (tùy nguồn tải).
*   **Các trường quan trọng (Fields):**
    *   `functionSource`: Mã nguồn C/C++ (String).
    *   `CWE-119`, `CWE-120`, ...: Các cột nhãn (Boolean/Int).
*   **Mapping Nhãn (Label Mapping):**
    *   0: CWE-119
    *   1: CWE-120
    *   2: CWE-469
    *   3: CWE-476
    *   4: CWE-other

## 2. Dữ liệu Đã xử lý (Processed Data)
Sau khi chạy qua bước tiền xử lý, dữ liệu nên được lưu dưới dạng **Python Pickle (.pkl)** hoặc **HuggingFace Dataset** để giữ nguyên các object Python.

### 2.1. Cấu trúc chung (List of Dicts)
Mỗi mẫu dữ liệu là một dictionary:
```python
{
    "code": "int main() { ... }",  # Mã nguồn gốc
    "label": 2,                      # Nhãn số nguyên (0-4)
    "tokens": ["int", "main", ...],  # List các token
}
```

### 2.2. Input cho từng Mô hình (Tensor Format)

Khi đi qua `Dataset.__getitem__`, dữ liệu sẽ chuyển thành Tensor:

**A. Cho CodeBERT & UniXcoder**
*   `input_ids`: Tensor[Long] - Shape `(max_len,)` - ID của token trong từ điển.
*   `attention_mask`: Tensor[Long] - Shape `(max_len,)` - 1 cho token thật, 0 cho padding.
*   `labels`: Tensor[Long] - Shape `(1,)` - Nhãn thực tế.

**B. Cho GraphCodeBERT (Đặc biệt)**
GraphCodeBERT yêu cầu thêm thông tin về Data Flow Graph (DFG).
*   `input_ids`: Như trên.
*   `attention_mask`: Như trên.
*   `position_ids`: Tensor[Long] - Shape `(max_len,)` - Vị trí của token trong cây (cần thiết để mô hình hiểu cấu trúc đồ thị).

## 3. Dữ liệu Vector Đặc trưng (Feature Vectors - cho Stacking)
Đầu ra của giai đoạn `extract_features` sẽ được lưu thành các file NumPy (`.npy`) để huấn luyện Meta-classifier.

**File Format:**
*   `train_features_codebert.npy`: Shape `(N_samples, 768)`
*   `train_features_graphcodebert.npy`: Shape `(N_samples, 768)`
*   `train_features_unixcoder.npy`: Shape `(N_samples, 768)`
*   `train_labels.npy`: Shape `(N_samples,)`

**Stacked Input (Đầu vào cho SVM/XGBoost):**
*   Khi nối (concatenate), shape sẽ là `(N_samples, 768 * 3)` = `(N_samples, 2304)`.

---

## Xem thêm

- **[Methodology](methodology.md)** - Cách các mô hình sử dụng dữ liệu này
- **[Technical Specification](technical_specification.md)** - Implementation của VulnerabilityDataset class
- **[Deployment Guide](deployment_guide.md)** - Hướng dẫn chuẩn bị và lưu trữ dữ liệu
