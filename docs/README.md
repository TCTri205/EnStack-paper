# EnStack: Một Framework Stacking Ensemble các Mô hình Ngôn ngữ Lớn để Nâng cao Khả năng Phát hiện Lỗ hổng trong Mã nguồn

**Tác giả:** Shahriyar Zaman Ridoy, Md. Shazzad Hossain Shaon, Alfredo Cuzzocrea, và Mst Shapna Akter

## Mục lục (Table of Contents)
- [Tóm tắt (Abstract)](#tóm-tắt-abstract)
- [Giới thiệu (Introduction)](#giới-thiệu-introduction)
- [Các đóng góp chính](#các-đóng-góp-chính)
- [Tài liệu liên quan](#tài-liệu-liên-quan)

---

## Tóm tắt (Abstract)
Việc tự động phát hiện các lỗ hổng phần mềm là rất quan trọng để tăng cường bảo mật. Bài báo này giới thiệu **EnStack**, một framework stacking ensemble mới giúp nâng cao khả năng phát hiện lỗ hổng bằng các kỹ thuật NLP. Phương pháp này kết hợp sức mạnh của nhiều mô hình ngôn ngữ lớn (LLM) đã được huấn luyện trước (pre-trained), chuyên về hiểu mã nguồn:
-   **CodeBERT** cho phân tích ngữ nghĩa.
-   **GraphCodeBERT** cho biểu diễn cấu trúc.
-   **UniXcoder** cho các khả năng đa phương thức (cross-modal).

Các mô hình này được tinh chỉnh (fine-tuned) trên bộ dữ liệu Draper VDISC. 
Đầu ra của chúng được tích hợp thông qua các bộ phân loại meta (meta-classifiers) 
như Logistic Regression, SVM, Random Forest, và XGBoost. EnStack nắm bắt hiệu quả 
các mẫu phức tạp mà các mô hình riêng lẻ có thể bỏ qua. Kết quả thực nghiệm cho thấy 
EnStack vượt trội hơn các phương pháp hiện có về độ chính xác (accuracy), 
độ chính xác (precision), độ nhạy (recall) và điểm F1.

## Giới thiệu (Introduction)
Các lỗ hổng phần mềm gây ra các mối đe dọa đáng kể, dẫn đến vi phạm dữ liệu 
và tổn thất tài chính. Các phương pháp truyền thống (kiểm tra thủ công, phân tích tĩnh) 
gặp khó khăn với sự phức tạp của phần mềm hiện đại.
Những tiến bộ gần đây trong AI, cụ thể là các LLM như CodeBERT và UniXcoder, 
cho thấy nhiều hứa hẹn nhưng thường chỉ tập trung vào các khía cạnh biểu diễn mã cụ thể 
(ngữ nghĩa hoặc cấu trúc). Việc sử dụng riêng lẻ có thể không nắm bắt được bản chất 
đa diện của các lỗ hổng.

**EnStack** giải quyết vấn đề này bằng cách kết hợp nhiều LLM thông qua 
kỹ thuật ensemble stacking để tận dụng các điểm mạnh riêng biệt nhằm tạo ra 
một hệ thống phát hiện mạnh mẽ hơn.

## Các đóng góp chính
1.  **Đề xuất Framework Stacking dựa trên Ensemble:** Tích hợp nhiều LLM đã được huấn luyện trước với các bộ phân loại meta.
2.  **Đánh giá toàn diện:** Đánh giá EnStack trên bộ dữ liệu Draper VDISC, 
    chứng minh hiệu năng vượt trội so với các mô hình riêng lẻ và các phương pháp 
    tiếp cận hiện có.
3.  **Nghiên cứu cắt giảm (Ablation Study):** Phân tích tác động của các kết hợp 
    mô hình và bộ phân loại meta khác nhau để định hướng cho các chiến lược ensemble 
    trong tương lai.

## Tài liệu liên quan

Để tìm hiểu chi tiết về dự án EnStack, vui lòng tham khảo các tài liệu sau:

- **[Phương pháp luận (Methodology)](methodology.md)** - Giải thích chi tiết về framework EnStack, 
  công thức toán học, và các mô hình cơ sở
- **[Đặc tả kỹ thuật (Technical Specification)](technical_specification.md)** - Cấu trúc mã nguồn, 
  thiết kế class, và các interface chính
- **[Cấu trúc dữ liệu (Data Schema)](data_schema.md)** - Định dạng dữ liệu đầu vào/đầu ra 
  và cấu trúc dataset
- **[Hướng dẫn triển khai (Deployment Guide)](deployment_guide.md)** - Quy trình triển khai 
  và cấu hình môi trường
- **[Thực nghiệm và Kết quả (Experiments and Results)](experiments_and_results.md)** - Chi tiết 
  thực nghiệm, hyperparameters, và phân tích kết quả
- **[Thảo luận và Kết luận (Conclusion)](conclusion.md)** - Thảo luận về kết quả, 
  hạn chế, và hướng phát triển tương lai
- **[Giải quyết sự cố (Troubleshooting)](TROUBLESHOOTING.md)** - Hướng dẫn khắc phục 
  các lỗi thường gặp
- **[Câu hỏi thường gặp (FAQ)](FAQ.md)** - Giải đáp các thắc mắc chung về dự án
- **[Cải tiến Checkpoint (Checkpoint Improvements)](checkpoint_improvements.md)** - Chi tiết về các cải tiến hệ thống lưu và xác minh checkpoint
- **[Hướng dẫn Xác minh Checkpoint (Checkpoint Verification Guide)](checkpoint_verification_guide.md)** - Hướng dẫn sử dụng các công cụ xác minh checkpoint

## 📊 Tóm tắt Kết quả (Results Summary)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| EnStack (Best) | **82.36%** | **82.28%** |
| UniXcoder | 81.54% | 81.49% |
| CodeBERT | 78.51% | 77.98% |

[Xem chi tiết kết quả →](experiments_and_results.md)
