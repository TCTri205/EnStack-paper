# Hướng dẫn Triển khai Dự án EnStack

## Mục lục (Table of Contents)
- [Mô hình Kiến trúc Triển khai](#1-mô-hình-kiến-trúc-triển-khai)
- [Cấu trúc Dự án Chuẩn](#2-cấu-trúc-dự-án-chuẩn)
- [Quy trình Làm việc](#3-quy-trình-làm-việc-workflow)
  - [Giai đoạn 1: Thiết lập](#giai-đoạn-1-thiết-lập-setup)
  - [Giai đoạn 2: Phát triển](#giai-đoạn-2-phát-triển-development-loop)
  - [Giai đoạn 3: Bàn giao](#giai-đoạn-3-bàn-giao-handover)
- [Hướng dẫn Chi tiết cho Notebook](#4-hướng-dẫn-chi-tiết-cho-notebook-chạy-main_pipelineipynb)

---

Tài liệu này hướng dẫn chi tiết quy trình triển khai, phát triển và bàn giao dự án tái tạo (reproduce) **EnStack Framework**. Quy trình được thiết kế theo tiêu chuẩn công nghiệp (Best Practices), đảm bảo tính chuyên nghiệp, dễ bảo trì và dễ dàng chuyển giao cho bên thứ ba.

## 1. Mô hình Kiến trúc Triển khai

Chúng ta sử dụng mô hình kết hợp **GitOps** và **Cloud Storage** để tối ưu hóa quy trình làm việc giữa môi trường phát triển cục bộ (Local) và môi trường thực thi (Google Colab).

*   **Source Code (Logic):** Được quản lý phiên bản chặt chẽ trên **GitHub**. Đây là "Single Source of Truth" cho logic của chương trình.
*   **Artifacts (Dữ liệu & Model):** Được lưu trữ trên **Google Drive** của người dùng. Google Drive đóng vai trò như một Persistent Storage (Ổ cứng bền vững) để lưu dataset nặng và các checkpoint model quý giá, tránh việc mất dữ liệu khi Colab reset phiên làm việc.
*   **Runtime (Môi trường thực thi):** **Google Colab** đóng vai trò là Compute Engine (Máy tính toán). Colab sẽ không lưu trữ code lâu dài mà sẽ "kéo" code mới nhất từ GitHub về mỗi khi khởi chạy.

**Sơ đồ luồng dữ liệu:**
```mermaid
graph LR
    Local[Local Machine (VS Code)] -- Push Code --> GitHub
    GitHub -- Pull Code --> Colab[Google Colab (Runtime)]
    Drive[Google Drive (Storage)] -- Mount Data/Models --> Colab
    Colab -- Save Checkpoints --> Drive
```

## 2. Cấu trúc Dự án Chuẩn

Dự án được tổ chức theo cấu trúc module hóa, tách biệt rõ ràng giữa cấu hình, mã nguồn xử lý, và sổ tay thực nghiệm.

```text
EnStack_Reproduction/
├── .gitignore               # Cấu hình bỏ qua các file rác, file dữ liệu lớn
├── README.md                # Tài liệu hướng dẫn sử dụng chính (Entry point)
├── requirements.txt         # Danh sách các thư viện phụ thuộc (Dependencies)
├── configs/                 # Quản lý cấu hình tập trung
│   └── config.yaml          # File chứa siêu tham số (Hyperparameters), đường dẫn
├── notebooks/               # Các file Jupyter Notebook dùng để chạy trên Colab
│   └── main_pipeline.ipynb  # Notebook chính: Setup -> Load Data -> Train -> Eval
├── src/                     # Mã nguồn lõi (Core Source Code)
│   ├── __init__.py
│   ├── data_loader.py       # Module xử lý dữ liệu: Load Draper VDISC, tạo DFG
│   ├── models.py            # Module định nghĩa kiến trúc: CodeBERT, UniXcoder...
│   ├── trainer.py           # Module quản lý quy trình huấn luyện (Training Loop)
│   └── utils.py             # Các hàm tiện ích: Load config, logging, metrics
└── scripts/                 # Các script hỗ trợ (Shell scripts)
    └── setup_env.sh         # Script cài đặt môi trường tự động (nếu cần)
```

## 3. Quy trình Làm việc (Workflow)

### Giai đoạn 1: Thiết lập (Setup)
1.  **Khởi tạo Repo:** Tạo repository trên GitHub.
2.  **Cấu trúc Local:** Tạo thư mục dự án trên máy cục bộ theo cấu trúc ở Mục 2.
3.  **Chuẩn bị Dữ liệu:**
    *   Tải bộ dữ liệu Draper VDISC.
    *   Upload lên Google Drive theo cấu trúc: `My Drive/EnStack_Data/raw/`.

### Giai đoạn 2: Phát triển (Development Loop)
1.  **Coding:** Lập trình các tính năng (data loading, model definition) trên IDE ở máy cục bộ (VS Code/PyCharm).
2.  **Commit & Push:** Khi hoàn thành một module hoặc sửa lỗi, push code lên GitHub.
    ```bash
    git add .
    git commit -m "Update training logic"
    git push origin main
    ```
3.  **Testing trên Colab:**
    *   Mở `notebooks/main_pipeline.ipynb` trên Google Colab.
    *   Chạy cell "Update Code" (sử dụng `git pull`) để lấy code mới nhất từ GitHub.
    *   Chạy thử nghiệm để kiểm tra lỗi hoặc kết quả.

### Giai đoạn 3: Bàn giao (Handover)
Khi bàn giao dự án cho người khác, chỉ cần cung cấp:
1.  **Link GitHub Repository.**
2.  **File `notebooks/main_pipeline.ipynb`** (hoặc hướng dẫn mở file này từ Repo).
3.  **Hướng dẫn chuẩn bị dữ liệu:** Yêu cầu người nhận tạo thư mục trên Drive của họ và upload dataset vào đó.

## 4. Hướng dẫn Chi tiết cho Notebook Chạy (main_pipeline.ipynb)

File notebook này là giao diện chính để tương tác với dự án. Nó cần được thiết kế gồm 3 phần chính:

**Phần 1: Khởi tạo Môi trường (Environment Setup)**
*   Tự động clone repository từ GitHub.
*   Cài đặt các thư viện từ `requirements.txt`.
*   Cài đặt các thư viện đặc thù hệ thống (như `tree-sitter`).

**Phần 2: Kết nối Dữ liệu (Data Connection)**
*   Mount Google Drive.
*   Cho phép người dùng định nghĩa đường dẫn đến thư mục dữ liệu trên Drive của họ.

**Phần 3: Thực thi Pipeline (Execution)**
*   Load cấu hình từ `configs/config.yaml`.
*   Import các module từ `src/`.
*   Khởi tạo class `EnStackTrainer` và gọi hàm `train()`.

## 5. Troubleshooting và Lưu ý

### 5.1. Các lỗi thường gặp

**Lỗi 1: Git Pull thất bại**
```bash
# Giải pháp: Kiểm tra kết nối và reset về phiên bản mới nhất
!git fetch origin
!git reset --hard origin/main
```

**Lỗi 2: Out of Memory (OOM) trên Colab**
```python
# Giải pháp: Giảm batch size trong config.yaml
batch_size: 8  # Thay vì 16
```

**Lỗi 3: Module không tìm thấy**
```python
# Giải pháp: Thêm đường dẫn src vào PYTHONPATH
import sys
sys.path.append('/content/EnStack_Reproduction/src')
```

### 5.2. Best Practices

1.  **Checkpoint thường xuyên:** Lưu model checkpoint sau mỗi epoch để tránh mất dữ liệu.
2.  **Versioning:** Sử dụng git tags để đánh dấu các phiên bản quan trọng:
    ```bash
    git tag -a v1.0 -m "First stable version"
    git push origin v1.0
    ```
3.  **Backup dữ liệu:** Định kỳ sao lưu thư mục `EnStack_Data/` trên Google Drive.
4.  **Monitoring:** Sử dụng TensorBoard hoặc Weights & Biases để theo dõi quá trình huấn luyện.

### 5.3. Biến môi trường quan trọng

Trong file `configs/config.yaml`, các tham số quan trọng cần cấu hình:

```yaml
# Đường dẫn dữ liệu
data:
  root_dir: "/content/drive/MyDrive/EnStack_Data"
  train_file: "train_processed.pkl"
  val_file: "val_processed.pkl"
  test_file: "test_processed.pkl"

# Hyperparameters
training:
  batch_size: 16
  epochs: 10
  learning_rate: 2e-5
  max_length: 512

# Model configuration
model:
  base_models: ["codebert", "graphcodebert", "unixcoder"]
  meta_classifier: "svm"  # Options: lr, svm, rf, xgboost
  num_labels: 5
```

## 6. Bảo mật và Quyền truy cập

-   **Không commit credentials:** Đảm bảo `.gitignore` bao gồm các file nhạy cảm (API keys, tokens).
-   **Private Repository:** Nếu dự án chứa dữ liệu nhạy cảm, giữ repository ở chế độ private.
-   **Google Drive Permissions:** Chỉ chia sẻ thư mục dữ liệu với những người cần thiết.

---

*Tài liệu này đóng vai trò là bản thiết kế kỹ thuật (Technical Blueprint) cho việc triển khai dự án EnStack. 
Để biết thêm chi tiết về cấu trúc mã nguồn, vui lòng tham khảo [Technical Specification](technical_specification.md). 
Để hiểu về phương pháp luận, xem [Methodology](methodology.md).*
