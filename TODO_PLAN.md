# Kế hoạch Triển khai Dự án EnStack

Tài liệu này theo dõi tiến độ xây dựng và hoàn thiện dự án EnStack từ đầu đến cuối, đảm bảo tính đầy đủ, chi tiết và tuân thủ các quy chuẩn đã đề ra.

## Giai đoạn 1: Khởi tạo Hạ tầng (Scaffolding)
*Mục tiêu: Thiết lập khung dự án chuẩn, đảm bảo tính tương thích giữa Local và Colab.*

- [ ] **1.1. Cấu trúc thư mục dự án**
    - [ ] Tạo thư mục `src/` (Mã nguồn lõi)
    - [ ] Tạo thư mục `configs/` (Cấu hình)
    - [ ] Tạo thư mục `notebooks/` (Giao diện chạy trên Colab)
    - [ ] Tạo thư mục `tests/` (Kiểm thử)
    - [ ] Tạo thư mục `scripts/` (Scripts hỗ trợ)
    - [ ] Tạo thư mục `docs/` (Tài liệu - Đã có, cần rà soát)

- [ ] **1.2. Quản lý Môi trường & Dependencies**
    - [ ] Tạo `requirements.txt`:
        - `torch>=1.10.0`
        - `transformers>=4.20.0`
        - `scikit-learn>=1.0`
        - `pandas>=1.3`
        - `tree-sitter>=0.20`
        - `tqdm`, `pyyaml`, `pytest`, `black`, `ruff`
    - [ ] Tạo `.gitignore` chuẩn cho Python/AI project:
        - Ignore `data/`, `*.pkl`, `*.h5` (Dữ liệu lớn)
        - Ignore `*.bin`, `*.pth`, `checkpoints/` (Model weights)
        - Ignore `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`
        - Ignore `.env` (Secrets)

## Giai đoạn 2: Cấu hình Hệ thống (Configuration)
*Mục tiêu: Tách biệt mã nguồn và tham số (Configuration Management).*

- [ ] **2.1. File cấu hình trung tâm (`configs/config.yaml`)**
    - [ ] **Data Config:** Định nghĩa đường dẫn `root_dir` (ưu tiên path Google Drive), tên file train/val/test.
    - [ ] **Model Config:** Danh sách `base_models` (codebert, graphcodebert, unixcoder), `meta_classifier`.
    - [ ] **Training Config:** `batch_size` (16), `epochs` (10), `learning_rate` (2e-5), `max_length` (512), `seed` (42).

- [ ] **2.2. Module tiện ích (`src/utils.py`)**
    - [ ] Hàm `load_config(path)`: Đọc file YAML và trả về Dict/Object.
    - [ ] Hàm `setup_logging()`: Cấu hình logger để ghi log ra console và file (thay vì print).
    - [ ] Hàm `set_seed(seed)`: Đảm bảo tính tái lập (reproducibility).

## Giai đoạn 3: Phát triển Mã nguồn Lõi (Core Implementation)
*Mục tiêu: Hiện thực hóa logic theo Technical Specification và Methodology.*

- [ ] **3.1. Xử lý Dữ liệu (`src/dataset.py`)**
    - [ ] Class `VulnerabilityDataset(Dataset)`:
        - `__init__`: Load data, khởi tạo tokenizer.
        - `__getitem__`: Tokenize text, xử lý padding/truncation.
        - **Logic đặc biệt cho GraphCodeBERT:** Tạo `position_ids` (nếu cần thiết và khả thi trong phạm vi hiện tại) hoặc xử lý input format đặc thù.
    - [ ] Hàm `create_dataloaders`: Tạo Train/Val/Test loaders.

- [ ] **3.2. Mô hình (`src/models.py`)**
    - [ ] Class `EnStackModel(nn.Module)`:
        - Wrapper cho `RobertaForSequenceClassification`.
        - Hỗ trợ load pre-trained weights từ HuggingFace (`microsoft/codebert-base`, `microsoft/graphcodebert-base`, `microsoft/unixcoder-base`).
        - Hàm `forward`: Trả về logits và loss.
        - Hàm `get_embedding`: Trả về vector đặc trưng (CLS token) cho Stacking.

- [ ] **3.3. Quy trình Huấn luyện (`src/trainer.py`)**
    - [ ] Class `EnStackTrainer`:
        - `__init__`: Nhận model, dataloaders, optimizer, scheduler.
        - `train_epoch()`: Vòng lặp train một epoch.
        - `evaluate()`: Tính toán Loss, Accuracy, F1.
        - `save_checkpoint()`: Lưu model vào đường dẫn cấu hình (Drive).
        - `extract_features()`: Chạy inference để lấy vector cho Stacking.

## Giai đoạn 4: Stacking Ensemble (Meta-Classifier)
*Mục tiêu: Kết hợp các mô hình cơ sở để tăng độ chính xác.*

- [ ] **4.1. Module Stacking (`src/stacking.py` hoặc tích hợp trong `trainer.py`)**
    - [ ] Hàm `prepare_meta_features`: Tổng hợp vector từ 3 model cơ sở.
    - [ ] Hàm `train_meta_classifier`: Huấn luyện SVM/LR/XGBoost trên meta-features.
    - [ ] Hàm `evaluate_ensemble`: Đánh giá mô hình cuối cùng trên Test set.

## Giai đoạn 5: Tích hợp Google Colab & Deployment
*Mục tiêu: Đảm bảo quy trình chạy mượt mà trên Cloud.*

- [ ] **5.1. Notebook chính (`notebooks/main_pipeline.ipynb`)**
    - [ ] Setup cell: Mount Drive, `git pull`, cài requirements.
    - [ ] Config cell: Override config paths cho môi trường Colab.
    - [ ] Pipeline cell: Chạy Training -> Extraction -> Stacking -> Evaluation.
    - [ ] Visualization cell: Vẽ biểu đồ Loss/Accuracy (nếu cần).

- [ ] **5.2. Script hỗ trợ (`scripts/`)**
    - [ ] `setup_colab.sh`: Script bash để tự động cài đặt môi trường (nếu notebook quá dài).

## Giai đoạn 6: Kiểm thử & Đảm bảo Chất lượng (QA)
*Mục tiêu: Đảm bảo code chạy đúng và ổn định.*

- [ ] **6.1. Unit Tests (`tests/`)**
    - [ ] `test_dataset.py`: Kiểm tra shape của tensors đầu ra.
    - [ ] `test_models.py`: Kiểm tra forward pass (fake input) không bị lỗi dimension.
    - [ ] `test_trainer.py`: Chạy thử 1 step training (smoke test).

- [ ] **6.2. Code Quality**
    - [ ] Chạy `ruff check .` để tìm lỗi linter.
    - [ ] Chạy `black .` để format code.
    - [ ] Chạy `mypy .` để kiểm tra type safety.

## Giai đoạn 7: Tài liệu & Bàn giao
*Mục tiêu: Hoàn thiện hồ sơ dự án.*

- [ ] **7.1. Cập nhật README.md**
    - [ ] Hướng dẫn cài đặt nhanh.
    - [ ] Hướng dẫn chạy trên Colab (link tới notebook).
- [ ] **7.2. Rà soát tài liệu docs/**
    - [ ] Đảm bảo các file hướng dẫn khớp với code thực tế.

---
*Trạng thái: Đang khởi tạo*
