# Phương pháp luận (Methodology)

> **Xem thêm:** 
> - [Technical Specification](technical_specification.md) - Cấu trúc mã nguồn chi tiết
> - [Experiments and Results](experiments_and_results.md) - Kết quả thực nghiệm
> - [Data Schema](data_schema.md) - Định dạng dữ liệu

---

## Công thức bài toán
Mục tiêu là dự đoán lớp lỗ hổng $\hat{y}_i$ cho một đoạn mã $x_i$ dựa trên tập dữ liệu $D = \{(x_i, y_i)\}_{i=1}^n$, trong đó $y_i$ đại diện cho nhãn lớp lỗ hổng (các danh mục CWE).
EnStack sử dụng một tập hợp các mô hình cơ sở $\mathcal{M} = \{M_k\}_{k=1}^K$ (ví dụ: CodeBERT, GraphCodeBERT, UniXcoder) đã được tinh chỉnh trên dữ liệu huấn luyện.
Một vector đặc trưng meta (meta-feature vector) $z_i$ được xây dựng bằng cách nối các đầu ra của tất cả các mô hình cơ sở:
$$ z_i = [M_1(x_i), M_2(x_i), \dots, M_K(x_i)] $$
Một bộ phân loại meta $\mathcal{F}_{meta}$ sau đó được huấn luyện trên các đặc trưng này để dự đoán $\hat{y}_i$.

## Framework EnStack
Framework tuân theo các bước sau (Thuật toán 1):

**Thuật toán 1: EnStack Training Pipeline**
```
Input:  D_train, D_val, D_test (Training, Validation, Test datasets)
        M = {M₁, M₂, ..., Mₖ} (Base models: CodeBERT, GraphCodeBERT, UniXcoder)
        F_candidates = {LR, RF, SVM, XGBoost} (Meta-classifier candidates)

Output: Trained EnStack model with optimal meta-classifier

1:  # Chuẩn bị dữ liệu
2:  D_train ← Downsample(D_train)  # Cân bằng dữ liệu
3:  
4:  # Huấn luyện các mô hình cơ sở
5:  for each Mₖ in M do
6:      Mₖ ← FineTune(Mₖ, D_train)  # Fine-tune trên tập huấn luyện
7:  end for
8:  
9:  # Tạo Meta-features cho tập Validation
10: for each sample xⱼ in D_val do
11:     zⱼ ← [M₁(xⱼ), M₂(xⱼ), ..., Mₖ(xⱼ)]  # Nối các dự đoán
12: end for
13: Z_val ← {z₁, z₂, ..., z|D_val|}
14: 
15: # Huấn luyện và lựa chọn Meta-classifier
16: best_score ← 0
17: best_classifier ← None
18: for each F in F_candidates do
19:     F ← Train(F, Z_val, labels_val)
20:     score ← Evaluate(F, Z_val, labels_val)
21:     if score > best_score then
22:         best_score ← score
23:         best_classifier ← F
24:     end if
25: end for
26: 
27: # Đánh giá trên tập Test
28: for each sample xᵢ in D_test do
29:     zᵢ ← [M₁(xᵢ), M₂(xᵢ), ..., Mₖ(xᵢ)]
30:     ŷᵢ ← best_classifier(zᵢ)
31: end for
32: 
33: return best_classifier, {M₁, M₂, ..., Mₖ}
```

Chi tiết các bước:
1.  **Chuẩn bị dữ liệu:** Chia dữ liệu thành các tập Huấn luyện (Train), Kiểm định (Validation), và Kiểm tra (Test). Cân bằng dữ liệu huấn luyện thông qua kỹ thuật downsampling.
2.  **Huấn luyện mô hình cơ sở:** Tinh chỉnh (fine-tune) từng mô hình cơ sở ($M_k$) trên tập huấn luyện.
3.  **Tạo Meta-feature:** Đối với mỗi mẫu trong tập kiểm định, tính toán vector meta-feature $z_j$ bằng cách tổng hợp các dự đoán từ các mô hình cơ sở.
4.  **Huấn luyện & Lựa chọn Meta-classifier:** Huấn luyện các bộ phân loại meta ứng viên (LR, RF, SVM, XGBoost) trên các meta-feature kiểm định. Chọn bộ phân loại tối ưu dựa trên hiệu năng kiểm định.
5.  **Đánh giá:** Áp dụng framework đã chọn lên tập kiểm tra (test set).

## Các mô hình cơ sở (Base Models)
Framework sử dụng ba mô hình đã được huấn luyện trước, mỗi mô hình xử lý các khía cạnh khác nhau của mã nguồn:
1.  **CodeBERT:** Nắm bắt ý nghĩa ngữ nghĩa của các token mã (Ngôn ngữ tự nhiên + Ngôn ngữ lập trình).
2.  **GraphCodeBERT:** Nhấn mạnh các mối quan hệ cấu trúc thông qua đồ thị luồng dữ liệu (data flow graphs).
3.  **UniXcoder:** Thống nhất các biểu diễn đa phương thức (cú pháp và ngữ nghĩa).

Tất cả các mô hình đều được fine-tune sử dụng hàm mất mát Cross Entropy trên bộ dữ liệu Draper VDISC đã được downsample.

## Các bộ phân loại Meta (Meta-Classifiers)
Bốn bộ phân loại meta được sử dụng để xếp chồng (stack) các đầu ra của LLM:
-   **Logistic Regression (LR)** - Hồi quy Logistic
-   **Random Forest (RF)** - Rừng ngẫu nhiên
-   **Support Vector Machine (SVM)** (nhân RBF)
-   **XGBoost**

## Dữ liệu và Tiền xử lý
**Bộ dữ liệu Draper VDISC:**
-   Hơn 1.27 triệu hàm mã nguồn.
-   Các lớp được ánh xạ tới danh mục CWE:
    -   CWE-119 (Memory - Bộ nhớ)
    -   CWE-120 (Buffer Overflow - Tràn bộ nhớ đệm)
    -   CWE-469 (Integer Overflow - Tràn số nguyên)
    -   CWE-476 (Null Pointer - Con trỏ null)
    -   CWE-other (Các lỗi khác)
-   **Tiền xử lý:** Loại bỏ các mục null.
-   **Cân bằng:** Sử dụng Downsampling để giải quyết tình trạng mất cân bằng lớp nghiêm trọng (đặc biệt là các lớp thiểu số như CWE-469) nhằm ngăn chặn overfitting và nhiễu từ việc tạo dữ liệu tổng hợp.
