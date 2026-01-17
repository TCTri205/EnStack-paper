# Thực nghiệm và Kết quả

> **Liên quan:**
> - [Methodology](methodology.md) - Phương pháp luận và thuật toán
> - [Conclusion](conclusion.md) - Thảo luận về kết quả
> - [FAQ](FAQ.md) - Câu hỏi về reproducibility và cải thiện kết quả

---

## Mục lục (Table of Contents)
- [Thiết lập thực nghiệm](#thiết-lập-thực-nghiệm)
  - [Phần cứng và Phần mềm](#phần-cứng-và-phần-mềm)
  - [Chi tiết Tham số Huấn luyện](#chi-tiết-tham-số-huấn-luyện-hyperparameters)
- [Các mô hình cơ sở so sánh](#các-mô-hình-cơ-sở-so-sánh-baselines)
- [Các chỉ số đánh giá](#các-chỉ-số-đánh-giá)
- [Kết quả và Phân tích](#kết-quả-và-phân-tích)
  - [Hiệu năng mô hình](#hiệu-năng-mô-hình-quantitative-results)
  - [Phân tích trực quan hóa t-SNE](#phân-tích-trực-quan-hóa-t-sne-qualitative-analysis)
- [Nghiên cứu cắt giảm](#nghiên-cứu-cắt-giảm-ablation-study)

---

## Thiết lập thực nghiệm

### Dữ liệu và Phân bố (Data Distribution)
Bộ dữ liệu Draper VDISC được chia thành các tập Training (80%), Validation (10%), và Test (10%). Do sự mất cân bằng dữ liệu nghiêm trọng, kỹ thuật downsampling đã được áp dụng.

**Bảng: Phân bố dữ liệu theo lớp (Dựa trên Table I)**

| Class Label | CWE Type | Training Samples | Validation Samples | Test Samples |
|:-----------:|:---------|:----------------:|:------------------:|:------------:|
| 0 | CWE-119 (Memory) | 5942 | 1142 | 1142 |
| 1 | CWE-120 (Buffer Overflow) | 5777 | 1099 | 1099 |
| 2 | CWE-469 (Integer Overflow) | 249 | 53 | 53 |
| 3 | CWE-476 (Null Pointer) | 2755 | 535 | 535 |
| 4 | CWE-other | 5582 | 1071 | 1071 |
| **Total** | | **20305** | **3900** | **3900** |

### Phần cứng và Phần mềm
-   **Phần cứng:** GPU NVIDIA Tesla P100.
-   **Phần mềm:** Framework PyTorch cho deep learning, thư viện Hugging Face Transformers 
    cho các mô hình ngôn ngữ lớn (LLM), và thư viện scikit-learn để triển khai 
    các kỹ thuật stacking ensemble (meta-classifiers).

### Chi tiết Tham số Huấn luyện (Hyperparameters)
Quá trình huấn luyện được chia thành hai giai đoạn: tinh chỉnh (fine-tuning) các mô hình cơ sở và huấn luyện các bộ phân loại meta.

**1. Các mô hình Transformer (CodeBERT, GraphCodeBERT, UniXcoder):**
Các tham số này được giữ nhất quán để đảm bảo tính công bằng khi so sánh:
-   **Batch Size (Kích thước lô):** 16
-   **Epochs (Số vòng lặp):** 10
-   **Learning Rate (Tốc độ học):** $2 \times 10^{-5}$
-   **Optimizer (Bộ tối ưu hóa):** AdamW
-   **Max Token Length (Độ dài token tối đa):** 512

**2. Các Bộ phân loại Meta (Meta-classifiers):**
-   **Logistic Regression (Hồi quy Logistic):**
    -   Max Iterations: 200
    -   Solver: liblinear
-   **Random Forest (Rừng ngẫu nhiên):**
    -   Number of Estimators (Số lượng cây): 200
    -   Max Depth (Độ sâu tối đa): 10
-   **Support Vector Machine (SVM):**
    -   Kernel: RBF (Radial Basis Function)
    -   Probability Estimation: True
    -   Random State: 42
-   **XGBoost:**
    -   Number of Estimators: 100
    -   Eval Metric: mlogloss
    -   Learning Rate: 0.1
    -   Max Depth: 6

## Các mô hình cơ sở so sánh (Baselines)
-   **Transformers riêng lẻ:** CodeBERT, GraphCodeBERT, UniXcoder.
-   **Non-transformer:** Attention LSTM (để làm nổi bật lợi ích của các mô hình ngôn ngữ đã được huấn luyện trước).

## Các chỉ số đánh giá
-   **Accuracy (Độ chính xác):** Tỷ lệ dự đoán đúng.
-   **Precision, Recall, F1-Score:** Đánh giá độ chính xác của các dự đoán tích cực và khả năng bao phủ của mô hình.
-   **AUC-Score:** Diện tích dưới đường cong ROC, đánh giá khả năng phân biệt giữa các lớp lỗ hổng.

## Kết quả và Phân tích

### Hiệu năng mô hình (Quantitative Results)

#### Bảng 1: So sánh hiệu năng các mô hình (Dựa trên Table III của bài báo)

| Mô hình | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---------|--------------|---------------|------------|--------------|---------|
| Attention LSTM | 73.00 | 72.97 | 73.00 | 72.95 | 77.54 |
| CodeBERT (C) | 78.51 | 77.85 | 78.51 | 77.98 | 92.16 |
| GraphCodeBERT (G) | 80.05 | 79.92 | 80.05 | 79.86 | 93.36 |
| UniXcoder (U) | 81.54 | 81.96 | 81.54 | 81.49 | **93.80** |
| **EnStack (G+U+SVM)** | **82.36** | **82.85** | **82.36** | **82.28** | 90.53 |
| **EnStack (G+U+LR)** | **82.36** | 82.59 | **82.36** | 82.21 | 92.85 |

**Phân tích chi tiết:**

-   **Mô hình riêng lẻ:**
    -   **UniXcoder** dẫn đầu trong số các mô hình đơn lẻ (Accuracy: 81.54%, F1: 81.49%, 
        AUC: 93.80%), nhờ khả năng biểu diễn đa phương thức (cross-modal).
    -   **Attention LSTM** kém hơn đáng kể (Accuracy: 73.00%), cho thấy hạn chế 
        của các mô hình không phải transformer trong việc nắm bắt ngữ nghĩa mã phức tạp.
-   **Ensemble Stacking (EnStack):**
    -   Việc kết hợp nhiều LLM mang lại hiệu suất vượt trội.
    -   **Cấu hình tốt nhất về Accuracy/F1:** Stacking **GraphCodeBERT + UniXcoder (G+U)** sử dụng **SVM** làm meta-classifier.
        -   **Accuracy:** **82.36%** (Cao nhất)
        -   **F1-Score:** **82.28%**
        -   **Precision:** **82.85%**
    -   **Khả năng phân biệt tốt nhất (AUC):** Cặp G+U với **Logistic Regression** đạt **AUC-Score 92.85%**, tuy nhiên vẫn thấp hơn UniXcoder đơn lẻ (93.80%) về chỉ số này.

#### Bảng 2: Ablation Study - So sánh các cấu hình EnStack (Trích xuất từ Table III)

| Kết hợp Mô hình | Meta-Classifier | Accuracy (%) | F1-Score (%) | AUC (%) |
|-----------------|-----------------|--------------|--------------|---------|
| C+G (CodeBERT + GraphCodeBERT) | SVM | 81.46 | 81.40 | 89.96 |
| C+G | Logistic Regression | 81.13 | 80.90 | 92.93 |
| C+G | Random Forest | 81.56 | 81.44 | 92.32 |
| C+G | XGBoost | 80.28 | 80.06 | 91.31 |
| **G+U (GraphCodeBERT + UniXcoder)** | **SVM** | **82.36** | **82.28** | 90.53 |
| **G+U** | **Logistic Regression** | **82.36** | 82.21 | **92.85** |
| G+U | Random Forest | 82.28 | 82.13 | 92.45 |
| G+U | XGBoost | 80.67 | 80.46 | 92.28 |

**Insights từ Ablation Study:**
-   Cặp **G+U** (GraphCodeBERT + UniXcoder) luôn cho kết quả tốt nhất.
-   **SVM** đạt F1-score cao nhất (82.28%), trong khi **Logistic Regression** đạt AUC cao hơn (92.85%) trong nhóm stacking G+U.


Kết quả này khẳng định rằng việc kết hợp cái nhìn sâu sắc về cấu trúc 
(từ GraphCodeBERT) với sự hiểu biết cú pháp-ngữ nghĩa (từ UniXcoder) 
tạo ra một bộ đặc trưng toàn diện nhất.

### Phân tích trực quan hóa t-SNE (Qualitative Analysis)
Bài báo sử dụng kỹ thuật t-SNE để trực quan hóa không gian biểu diễn (latent space) 
của các vector đặc trưng, so sánh giữa mô hình đơn lẻ và mô hình EnStack:

#### Hình 1: So sánh không gian biểu diễn

![t-SNE visualization của CodeBERT](images/tsne_codebert.png)
*Hình 1a: Không gian biểu diễn của CodeBERT (baseline) - Chú ý sự chồng chéo giữa các cụm*

![t-SNE visualization của EnStack](images/tsne_enstack.png)
*Hình 1b: Không gian biểu diễn của EnStack - Các cụm tách biệt rõ ràng hơn*

1.  **Mô hình cơ sở (CodeBERT):**
    -   Biểu đồ cho thấy **sự chồng chéo đáng kể (class overlap)** giữa các cụm màu 
        đại diện cho các loại lỗ hổng khác nhau.
    -   Điều này giải thích tại sao mô hình đơn lẻ có tỷ lệ dương tính giả cao hơn; 
        nó gặp khó khăn trong việc vẽ ranh giới rõ ràng giữa các loại lỗ hổng 
        có đặc điểm tương tự.

2.  **Mô hình EnStack (CodeBERT + GraphCodeBERT + Logistic Regression):**
    -   Biểu đồ hiển thị sự cải thiện rõ rệt về **cấu trúc cụm (cluster formation)**.
    -   Các điểm dữ liệu cùng loại lỗ hổng tụ lại chặt chẽ hơn và 
        **tách biệt tốt hơn (class separability)** với các nhóm khác.
    -   **Ý nghĩa:** Kỹ thuật stacking đã giúp mô hình học được các biểu diễn đặc trưng 
        tinh tế hơn, đẩy các lớp lỗ hổng ra xa nhau trong không gian vector, 
        giúp bộ phân loại meta dễ dàng đưa ra quyết định chính xác.

## Nghiên cứu cắt giảm (Ablation Study)
-   **Kết hợp mô hình:** Cặp **G+U** (GraphCodeBERT + UniXcoder) luôn cho kết quả 
    tốt hơn các kết hợp khác (như C+G), bất kể meta-classifier nào được sử dụng.
-   **Lựa chọn Meta-classifier:** Các bộ phân loại tuyến tính đơn giản 
    (**SVM, Logistic Regression**) hoạt động hiệu quả hơn các mô hình phức tạp 
    như XGBoost trong tầng stacking này. XGBoost có xu hướng kém hiệu quả hơn 
    (Accuracy 80.67% khi stack G+U), có thể do làm phức tạp hóa vấn đề không cần thiết 
    trên tập đặc trưng đã được trích xuất tốt từ LLM.
