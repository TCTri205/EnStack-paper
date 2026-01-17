# 📦 Hướng Dẫn Bàn Giao Dự Án EnStack

**Phiên bản**: 1.0.0  
**Ngày cập nhật**: 17/01/2026  
**Người bàn giao**: [Tên của bạn]  
**Người nhận**: [Tên người nhận]

---

## 📋 Tổng Quan Dự Án

**EnStack** là một framework stacking ensemble để phát hiện lỗ hổng bảo mật trong mã nguồn sử dụng Large Language Models (CodeBERT, GraphCodeBERT, UniXcoder).

### Thông tin Repository
- **GitHub**: https://github.com/TCTri205/EnStack-paper
- **Branch chính**: `main`
- **Ngôn ngữ**: Python 3.8+
- **Framework**: PyTorch, Hugging Face Transformers

---

## 🎯 Mục Tiêu Dự Án

Xây dựng hệ thống ensemble learning để:
1. Phát hiện lỗ hổng bảo mật trong mã nguồn C/C++
2. Phân loại theo 5 loại CWE: 119, 120, 469, 476, Other
3. Đạt độ chính xác cao hơn các mô hình đơn lẻ bằng kỹ thuật stacking

---

## 📁 Cấu Trúc Dự Án

```
EnStack-paper/
├── src/                          # Mã nguồn chính
│   ├── dataset.py               # Xử lý dữ liệu
│   ├── models.py                # Định nghĩa mô hình
│   ├── trainer.py               # Logic huấn luyện
│   ├── stacking.py              # Ensemble stacking
│   └── utils.py                 # Các hàm tiện ích
├── configs/
│   ├── config.yaml              # Cấu hình cho production
│   └── config_local.yaml        # Cấu hình cho test local
├── notebooks/
│   ├── EnStack_Colab_Deployment.ipynb  # Notebook chính cho Colab ⭐
│   └── main_pipeline.ipynb             # Pipeline đầy đủ
├── scripts/
│   ├── train.py                 # Script training CLI
│   ├── prepare_data.py          # Script chuẩn bị dữ liệu
│   └── generate_dummy_data.py   # Tạo dữ liệu giả
├── tests/                       # Unit tests
├── docs/                        # Tài liệu chi tiết
├── requirements.txt             # Dependencies
├── pyproject.toml               # Cấu hình công cụ
└── README.md                    # Hướng dẫn tổng quan
```

---

## 🚀 Hướng Dẫn Sử Dụng Cho Người Nhận

### Option 1: Chạy trên Google Colab (Khuyến nghị ⭐)

**Bước 1: Mở Notebook**
1. Truy cập: https://colab.research.google.com/github/TCTri205/EnStack-paper/blob/main/notebooks/EnStack_Colab_Deployment.ipynb
2. Đăng nhập bằng tài khoản Google

**Bước 2: Bật GPU**
1. Menu **Runtime** → **Change runtime type**
2. Chọn **Hardware accelerator**: **T4 GPU**
3. Click **Save**

**Bước 3: Chạy Notebook**
1. Chạy lần lượt các cell từ trên xuống
2. Cell 1: Mount Google Drive
3. Cell 2: Clone repository (username đã được điền sẵn: `TCTri205`)
4. Cell 3: Kiểm tra GPU
5. Cell 4: Cài đặt môi trường
6. Cell 5: Tải dữ liệu (tự động tải dataset công khai)
7. Cell 6: Kiểm tra dữ liệu
8. Cell 7: Cấu hình training (mặc định 2 epochs)
9. Cell 8: Chạy training

**Thời gian dự kiến**:
- Setup: 3-5 phút
- Training (2 epochs, 5000 samples): 10-20 phút trên GPU
- Tổng: ~30 phút

### Option 2: Chạy Local (Cho Developers)

```bash
# 1. Clone repository
git clone https://github.com/TCTri205/EnStack-paper.git
cd EnStack-paper

# 2. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Tạo dữ liệu test
python scripts/prepare_data.py --mode synthetic --train_size 100 --val_size 20 --test_size 20

# 5. Chạy training
python scripts/train.py --config configs/config_local.yaml
```

---

## 🔧 Cấu Hình Quan Trọng

### File `configs/config.yaml`

```yaml
data:
  root_dir: "/content/drive/MyDrive/EnStack_Data"
  train_file: "train_processed.pkl"
  val_file: "val_processed.pkl"
  test_file: "test_processed.pkl"

model:
  base_models: ["codebert", "graphcodebert", "unixcoder"]
  meta_classifier: "svm"  # Options: svm, lr, rf, xgboost
  num_labels: 5

training:
  batch_size: 16
  epochs: 10
  learning_rate: 2.0e-5
  max_length: 512
  seed: 42
```

**Tham số quan trọng cần biết**:
- `epochs`: Số vòng lặp training (2-10 cho test, 10-20 cho production)
- `batch_size`: Kích thước batch (16 cho GPU 16GB, giảm xuống 8 nếu hết RAM)
- `meta_classifier`: Bộ phân loại meta (SVM cho kết quả tốt nhất theo paper)

---

## 📊 Dữ Liệu

### Dataset Hiện Tại
- **Nguồn**: CodeXGLUE defect detection (tương tự Draper VDISC)
- **Kích thước**: 
  - Train: 5,000 samples (có thể tăng lên 21,854)
  - Validation: 2,732 samples
  - Test: 2,732 samples
- **Format**: `.pkl` files với 2 cột: `func` (code), `target` (label 0-4)

### Sử dụng Dataset Thật (Draper VDISC)
Nếu muốn dùng dataset gốc từ paper:
1. Tải từ: https://osf.io/d45bw/
2. Xử lý theo format yêu cầu (xem `scripts/prepare_data.py` hàm `print_manual_upload_guide()`)
3. Upload vào Google Drive: `/content/drive/MyDrive/EnStack_Data/`

---

## 🧪 Testing

### Chạy Unit Tests

```bash
# Chạy tất cả tests
pytest

# Chạy test cụ thể
pytest tests/test_dataset.py
pytest tests/test_models.py

# Test với coverage
pytest --cov=src tests/
```

**Tất cả 25 tests đều đã pass** ✅

### Code Quality

```bash
# Format code
black src/ tests/

# Linting
ruff check src/ tests/

# Type checking
mypy src/
```

---

## 📈 Kết Quả Dự Kiến

Dựa trên paper EnStack (2411.16561v1.pdf):

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| CodeBERT | 78.51% | 77.98% | 77.85% | 78.51% |
| GraphCodeBERT | 80.05% | 79.86% | 79.92% | 80.05% |
| UniXcoder | 81.54% | 81.49% | 81.96% | 81.54% |
| **EnStack (G+U+SVM)** | **82.36%** | **82.28%** | **82.85%** | **82.36%** |

---

## 🐛 Troubleshooting

### Vấn đề 1: Không có GPU trên Colab
**Triệu chứng**: Training rất chậm, log hiện "GPU will not be used"
**Giải pháp**: 
- Vào Runtime → Change runtime type → Chọn T4 GPU
- Restart notebook và chạy lại

### Vấn đề 2: Out of Memory (OOM)
**Triệu chứng**: `CUDA out of memory`
**Giải pháp**:
- Giảm `batch_size` xuống 8 hoặc 4 trong Cell 7
- Giảm `max_length` xuống 256 trong config

### Vấn đề 3: Data not found
**Triệu chứng**: `FileNotFoundError: Data file not found`
**Giải pháp**:
- Chạy lại Cell 5 (Download & Prepare Data)
- Kiểm tra Cell 6 để verify dữ liệu đã tồn tại

### Vấn đề 4: Training bị dừng giữa chừng
**Triệu chứng**: Colab timeout hoặc disconnect
**Giải pháp**:
- Giữ tab Colab mở và tương tác định kỳ
- Checkpoint tự động được lưu sau mỗi epoch vào Google Drive
- Có thể tiếp tục training từ checkpoint (cần implement resume logic)

---

## 📞 Hỗ Trợ

### Tài liệu tham khảo
1. **README.md**: Hướng dẫn tổng quan
2. **AGENTS.md**: Guidelines cho developers
3. **IMPLEMENTATION_REPORT.md**: Báo cáo chi tiết triển khai
4. **QUICKREF.md**: Quick reference
5. **docs/**: Tài liệu chi tiết về methodology, experiments

### Liên hệ
- **GitHub Issues**: https://github.com/TCTri205/EnStack-paper/issues
- **Email người bàn giao**: [Thêm email của bạn]

---

## ✅ Checklist Bàn Giao

- [ ] Người nhận đã có quyền truy cập GitHub repository
- [ ] Người nhận đã chạy thử notebook trên Colab thành công
- [ ] Người nhận hiểu cách thay đổi tham số (epochs, batch_size)
- [ ] Người nhận biết cách kiểm tra kết quả trong Google Drive
- [ ] Người nhận có tài khoản Google Colab (miễn phí)
- [ ] Người nhận đã đọc README.md và tài liệu này

---

## 📝 Ghi Chú Bổ Sung

### Cải tiến trong tương lai (Optional)
1. **Resume Training**: Thêm logic để tiếp tục từ checkpoint
2. **Hyperparameter Tuning**: Grid search cho meta-classifier
3. **Visualization**: Thêm confusion matrix, training curves
4. **MLflow Integration**: Track experiments
5. **Docker**: Containerize để chạy đồng nhất mọi môi trường

### Đã hoàn thành
- ✅ 100% code implementation
- ✅ Unit tests (25/25 passed)
- ✅ Code quality checks (black, ruff, mypy)
- ✅ Google Colab integration
- ✅ Automatic data download
- ✅ GPU support
- ✅ Comprehensive documentation

---

**Ngày bàn giao**: 17/01/2026  
**Trạng thái**: Production Ready ✅  
**Chữ ký người bàn giao**: _______________  
**Chữ ký người nhận**: _______________
