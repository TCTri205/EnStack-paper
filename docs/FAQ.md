# Câu hỏi Thường gặp (FAQ - Frequently Asked Questions)

Tài liệu này tổng hợp các câu hỏi thường gặp về dự án EnStack Framework.

## Mục lục
- [Về Dự án](#về-dự-án)
- [Về Dữ liệu](#về-dữ-liệu)
- [Về Mô hình](#về-mô-hình)
- [Về Triển khai](#về-triển-khai)
- [Về Kết quả](#về-kết-quả)

---

## Về Dự án

### EnStack là gì?

EnStack là một framework stacking ensemble kết hợp nhiều mô hình ngôn ngữ lớn (LLMs) 
để phát hiện lỗ hổng bảo mật trong mã nguồn. Framework này tích hợp CodeBERT, 
GraphCodeBERT, và UniXcoder thông qua các meta-classifiers như SVM và Logistic Regression.

### Tại sao cần sử dụng ensemble thay vì mô hình đơn lẻ?

Mỗi mô hình có điểm mạnh riêng:
- **CodeBERT**: Tốt cho phân tích ngữ nghĩa
- **GraphCodeBERT**: Nắm bắt cấu trúc data flow
- **UniXcoder**: Biểu diễn đa phương thức (cross-modal)

Việc kết hợp giúp tận dụng tất cả các điểm mạnh này, đạt accuracy 82.36% 
so với 81.54% của mô hình đơn lẻ tốt nhất (UniXcoder).

### Dự án này phù hợp với ai?

- Nghiên cứu viên về AI Security
- Kỹ sư phát triển công cụ phân tích mã nguồn
- Sinh viên học về Machine Learning và Software Security
- Bất kỳ ai quan tâm đến việc phát hiện lỗ hổng tự động

---

## Về Dữ liệu

### Tôi có thể lấy dataset Draper VDISC ở đâu?

Dataset Draper VDISC có thể được tải từ:
- [NIST SARD](https://samate.nist.gov/SARD/)
- [Draper Laboratory GitHub](https://github.com/draperlaboratory/)

**Lưu ý:** Cần đăng ký và đồng ý với điều khoản sử dụng.

### Dataset có bao nhiêu mẫu?

- Tổng số: Hơn 1.27 triệu hàm mã nguồn C/C++
- Sau downsampling: ~200,000 mẫu (để cân bằng các class)
- Train/Val/Test split: 70%/15%/15%

### Dataset có hỗ trợ ngôn ngữ lập trình nào?

Hiện tại chỉ hỗ trợ **C và C++**. Các nghiên cứu tương lai có thể mở rộng 
sang Python, Java, JavaScript.

### Tại sao sử dụng downsampling thay vì oversampling?

Downsampling được chọn vì:
1. Tránh overfitting do dữ liệu tổng hợp
2. Giảm thời gian training
3. Các lớp thiểu số (như CWE-469) có quá ít mẫu để SMOTE hiệu quả

---

## Về Mô hình

### Tại sao chọn GraphCodeBERT + UniXcoder thay vì sử dụng cả 3 mô hình?

Ablation study cho thấy:
- **G+U (GraphCodeBERT + UniXcoder)**: Accuracy 82.36%
- **C+G+U (cả 3 mô hình)**: Accuracy 81.89%

Thêm CodeBERT không cải thiện hiệu năng mà còn tăng chi phí tính toán.

### Meta-classifier nào tốt nhất?

**SVM và Logistic Regression** cho kết quả tốt nhất:
- SVM: Accuracy 82.36%, F1 82.28% (tốt nhất về F1)
- LR: Accuracy 82.36%, F1 82.21%, AUC 92.85% (tốt nhất về AUC trong nhóm stacking)

XGBoost và Random Forest kém hơn vì có thể overfit trên meta-features.

### Thời gian training mất bao lâu?

**Trên GPU NVIDIA Tesla P100:**
- Fine-tune mỗi base model: ~6-8 giờ
- Train meta-classifier: ~10-15 phút
- **Tổng cộng**: ~20-25 giờ cho toàn bộ pipeline

**Trên Google Colab (Free tier):**
- Có thể mất 2-3 ngày do giới hạn GPU runtime

### Model có thể detect bao nhiêu loại lỗ hổng?

Hiện tại detect 5 categories (multi-class classification):
- CWE-119 (Memory corruption)
- CWE-120 (Buffer overflow)
- CWE-469 (Integer overflow)
- CWE-476 (Null pointer dereference)
- CWE-other (Các lỗi khác)

---

## Về Triển khai

### Cần những phần cứng gì để chạy dự án?

**Tối thiểu:**
- GPU: 12GB VRAM (ví dụ: Tesla T4, RTX 3060)
- RAM: 16GB
- Storage: 50GB

**Khuyến nghị:**
- GPU: 16GB+ VRAM (Tesla V100, A100)
- RAM: 32GB
- Storage: 100GB (để lưu checkpoints)

### Có thể chạy trên CPU không?

Có, nhưng **rất chậm**:
- Training trên CPU có thể mất 1-2 tuần
- Inference: ~10 giây/sample (so với <1 giây trên GPU)

**Khuyến nghị:** Sử dụng Google Colab (free GPU) nếu không có GPU local.

### Làm sao để sử dụng Google Colab hiệu quả?

1. Sử dụng **Colab Pro** ($9.99/tháng) để có:
   - Longer runtime (24h thay vì 12h)
   - Better GPUs (V100/A100 thay vì T4)
   - More RAM (32GB thay vì 12GB)

2. Lưu checkpoint thường xuyên lên Google Drive

3. Sử dụng script chống timeout (xem [Troubleshooting](TROUBLESHOOTING.md))

### Có cần biết Deep Learning không?

**Để sử dụng:** Không nhất thiết - chỉ cần follow hướng dẫn trong 
[Deployment Guide](deployment_guide.md)

**Để modify/improve:** Nên có kiến thức về:
- PyTorch basics
- Transformer architecture
- Ensemble learning

---

## Về Kết quả

### Kết quả trong paper có reproducible không?

**Có**, nhưng cần lưu ý:
- Sử dụng cùng phiên bản thư viện (xem `requirements.txt`)
- Cùng random seed (42)
- Cùng hyperparameters trong `config.yaml`
- Kết quả có thể dao động ±0.5% do random initialization

### Tại sao kết quả của tôi khác với trong paper?

Các nguyên nhân có thể:
1. **Phiên bản thư viện khác nhau**: Transformers 4.20 vs 4.30 có thể cho kết quả khác
2. **Random seed**: Thử chạy nhiều lần với seeds khác nhau
3. **Data split**: Verify train/val/test split có giống không
4. **Hardware**: GPU khác nhau có thể cho kết quả hơi khác do floating point precision

### Làm sao để cải thiện accuracy hơn nữa?

**Các hướng có thể thử:**
1. **Data augmentation**: Thêm dữ liệu từ dataset khác (e.g., Big-Vul)
2. **Hyperparameter tuning**: Grid search cho learning rate, batch size
3. **Thêm base models**: Thử CodeT5, CodeGen
4. **Advanced meta-classifiers**: Neural network-based meta-learner
5. **Few-shot learning**: Cải thiện cho rare classes (CWE-469)

### Có thể apply cho production system không?

**Có, nhưng cần:**
1. **Optimize inference speed**:
   - Model quantization (INT8)
   - ONNX runtime
   - Batch processing

2. **Handle edge cases**:
   - Code quá dài (>512 tokens)
   - Ngôn ngữ khác C/C++
   - Obfuscated code

3. **CI/CD integration**:
   - GitHub Actions workflow
   - Pre-commit hooks
   - API service

---

## Câu hỏi Nâng cao

### Có thể fine-tune trên custom dataset không?

**Có**, bạn cần:
1. Chuẩn bị dữ liệu theo format trong [Data Schema](data_schema.md)
2. Annotate labels cho các lỗ hổng
3. Modify `data_loader.py` để load custom dataset
4. Retrain base models và meta-classifier

### Làm sao để add thêm CWE categories mới?

```python
# 1. Update config
num_labels = 7  # Thay vì 5

# 2. Retrain base models với num_labels mới
model = EnStackModel("microsoft/codebert-base", num_labels=7)

# 3. Retrain meta-classifier
```

### EnStack có thể kết hợp với static analysis tools không?

**Có**, có thể tạo ensemble level 2:
- Level 1: EnStack (GraphCodeBERT + UniXcoder + SVM)
- Level 2: EnStack + Clang Static Analyzer + Coverity

Kết hợp có thể tăng precision cao hơn.

---

## Tài nguyên Bổ sung

- **Paper gốc**: [Link to paper]
- **GitHub Repository**: [Link to repo]
- **Hugging Face Models**: [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- **Dataset**: [Draper VDISC](https://osf.io/d45bw/)

## Liên hệ

Nếu có câu hỏi chưa được giải đáp:
- Mở issue trên GitHub
- Email: [your-email@example.com]
- Join Discord: [Link nếu có]

---

*Cập nhật lần cuối: 2026-01-16*
