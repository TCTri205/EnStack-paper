# Đặc tả Kỹ thuật (Technical Specification)

Tài liệu này định nghĩa cấu trúc mã nguồn, các lớp (Classes), và giao diện (Interfaces) 
chính cho dự án EnStack. Mục tiêu là đảm bảo tính nhất quán và mô-đun hóa cao cho mã nguồn.

> **Xem thêm:**
> - [Data Schema](data_schema.md) - Định dạng dữ liệu cho các class
> - [Deployment Guide](deployment_guide.md) - Hướng dẫn triển khai code
> - [Methodology](methodology.md) - Nguyên lý hoạt động của các module
> - [TROUBLESHOOTING](TROUBLESHOOTING.md) - Giải quyết lỗi khi implement

---

## 1. Cấu trúc Module (`src/`)

```text
src/
├── dataset.py       # Xử lý dữ liệu đầu vào
├── models.py        # Wrapper cho các mô hình Transformer
├── trainer.py       # Vòng lặp huấn luyện và đánh giá
└── utils.py         # Các hàm hỗ trợ
```

## 2. Chi tiết Class Design

### 2.1. Module `dataset.py`

**Class `VulnerabilityDataset(Dataset)`**
*   **Mục đích:** Kế thừa từ `torch.utils.data.Dataset`, chịu trách nhiệm load dữ liệu và tiền xử lý.
*   **Phương thức chính:**
    *   `__init__(file_path, tokenizer, model_type, max_len)`:
        *   `file_path`: Đường dẫn file dữ liệu đã xử lý (.pkl).
        *   `model_type`: 'codebert', 'graphcodebert', hoặc 'unixcoder'.
    *   `__getitem__(index)`: Trả về dictionary chứa tensors:
        *   `input_ids`
        *   `attention_mask`
        *   `position_ids` (cho GraphCodeBERT)
        *   `labels`
    *   `__len__()`: Trả về số lượng mẫu.

### 2.2. Module `models.py`

**Class `EnStackModel(nn.Module)`**
*   **Mục đích:** Một lớp vỏ bọc (wrapper) chung cho cả 3 loại mô hình, giúp việc training loop không cần thay đổi khi đổi mô hình.
*   **Phương thức chính:**
    *   `__init__(model_name, num_labels)`:
        *   Tự động tải `RobertaForSequenceClassification` từ Hugging Face dựa trên `model_name`.
    *   `forward(input_ids, attention_mask, labels=None)`:
        *   Trả về `logits` và `loss` (nếu có labels).
    *   `get_embedding(input_ids, attention_mask)`:
        *   Trả về vector đặc trưng (feature vector) từ lớp ẩn cuối cùng (thường là token `<s>` hoặc `[CLS]`).

### 2.3. Module `trainer.py`

**Class `EnStackTrainer`**
*   **Mục đích:** Quản lý toàn bộ vòng đời huấn luyện.
*   **Thuộc tính:**
    *   `model`: Instance của `EnStackModel`.
    *   `optimizer`: AdamW.
    *   `scheduler`: Linear warm-up.
*   **Phương thức chính:**
    *   `train(train_loader, val_loader, epochs)`: Chạy vòng lặp huấn luyện.
    *   `evaluate(loader)`: Trả về accuracy, F1, precision, recall.
    *   `extract_features(loader)`: Trả về danh sách vector đặc trưng cho giai đoạn Stacking.
    *   `save_model(path)`: Lưu checkpoint.

## 3. Quy chuẩn Thư viện (Dependencies)

Các phiên bản thư viện khuyến nghị (sẽ nằm trong `requirements.txt`):
*   `torch>=1.10.0`
*   `transformers>=4.20.0`
*   `scikit-learn>=1.0`
*   `pandas>=1.3`
*   `tree-sitter>=0.20` (Bắt buộc cho GraphCodeBERT)
*   `tqdm` (Thanh tiến trình)

## 4. Quy ước Coding (Convention)

### 4.1. Type Hinting
Sử dụng **Type Hinting** cho tất cả các hàm để đảm bảo tính rõ ràng và dễ bảo trì:

```python
def train(self, epochs: int) -> float:
    """Huấn luyện mô hình qua số epochs đã cho."""
    pass

def extract_features(self, loader: DataLoader) -> List[np.ndarray]:
    """Trích xuất vector đặc trưng từ data loader."""
    pass
```

### 4.2. Docstring Style
Sử dụng **Docstring** style Google hoặc NumPy cho tất cả functions và classes:

**Ví dụ Google Style:**
```python
def evaluate(self, loader: DataLoader) -> Dict[str, float]:
    """
    Đánh giá mô hình trên tập dữ liệu.
    
    Args:
        loader (DataLoader): Data loader chứa dữ liệu đánh giá.
    
    Returns:
        Dict[str, float]: Dictionary chứa các metrics (accuracy, F1, precision, recall).
    
    Example:
        >>> trainer = EnStackTrainer(model, optimizer)
        >>> metrics = trainer.evaluate(test_loader)
        >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
    """
    pass
```

### 4.3. Logging
Sử dụng **logging** thay vì `print()` để quản lý output tốt hơn:

```python
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sử dụng trong code
logger.info("Bắt đầu huấn luyện epoch 1")
logger.warning("Validation loss tăng - có thể overfitting")
logger.error("Không thể load checkpoint từ đường dẫn đã cho")
```

### 4.4. Naming Convention
-   **Classes:** PascalCase (VD: `EnStackModel`, `VulnerabilityDataset`)
-   **Functions/Methods:** snake_case (VD: `train_model`, `extract_features`)
-   **Constants:** UPPER_SNAKE_CASE (VD: `MAX_LENGTH`, `NUM_LABELS`)
-   **Private methods:** Bắt đầu bằng underscore (VD: `_validate_input`)

### 4.5. Error Handling
Sử dụng exception handling đúng cách:

```python
def load_checkpoint(self, path: str) -> None:
    """Load model checkpoint từ file."""
    try:
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        logger.info(f"Đã load checkpoint từ {path}")
    except FileNotFoundError:
        logger.error(f"File không tồn tại: {path}")
        raise
    except KeyError as e:
        logger.error(f"Checkpoint thiếu key: {e}")
        raise ValueError(f"Checkpoint không hợp lệ: {e}")
```
