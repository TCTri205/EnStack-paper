# Thảo luận và Kết luận

> **Liên quan:**
> - [Experiments and Results](experiments_and_results.md) - Chi tiết kết quả đạt được
> - [Methodology](methodology.md) - Phương pháp đạt được kết quả này
> - [FAQ](FAQ.md) - Câu hỏi thường gặp về kết quả và hạn chế

---

## Thảo luận
Các thực nghiệm chứng minh rằng **EnStack** cải thiện đáng kể khả năng phát hiện lỗ hổng so với các mô hình đơn lẻ.
-   **Sức mạnh bổ trợ:** Sự thành công của sự kết hợp giữa GraphCodeBERT + UniXcoder 
    làm nổi bật giá trị của việc tích hợp dữ liệu luồng mã cấu trúc (structural code flow) 
    với các biểu diễn ngữ pháp/ngữ nghĩa đa phương thức (cross-modal syntactic/semantic).
-   **Meta-classifiers:** Các bộ phân loại đơn giản, dễ giải thích như SVM và LR 
    đã tổng hợp hiệu quả các không gian đặc trưng đa chiều từ các LLM, thường vượt trội hơn 
    so với XGBoost phức tạp hơn trong bối cảnh stacking cụ thể này.

## Các hạn chế
1.  **Mất cân bằng dữ liệu:** Mặc dù đã sử dụng kỹ thuật downsampling, bộ dữ liệu 
    vẫn bị mất cân bằng nghiêm trọng (ví dụ: CWE-469 xuất hiện rất ít), ảnh hưởng 
    đến khả năng tổng quát hóa.
2.  **Tính đặc thù của dữ liệu:** Việc phụ thuộc vào bộ dữ liệu Draper VDISC làm hạn chế 
    khả năng áp dụng cho các ngôn ngữ và loại lỗ hổng cụ thể chỉ có trong bộ dữ liệu này.
3.  **Chi phí tính toán:** Phương pháp ensemble yêu cầu fine-tuning nhiều LLM 
    và huấn luyện các meta-classifiers, tiêu tốn nhiều tài nguyên và có thể hạn chế 
    khả năng mở rộng trong thời gian thực.

## Kết luận
EnStack đã kết hợp thành công CodeBERT, GraphCodeBERT và UniXcoder để nâng cao 
khả năng phát hiện lỗ hổng tự động. Bằng cách fine-tuning các mô hình này và sử dụng 
kỹ thuật stacking với SVM/LR, framework đạt được kết quả tiên tiến nhất 
(Độ chính xác: 82.36%, AUC: 92.85%). Công trình này nhấn mạnh tiềm năng của các 
chiến lược ensemble LLM trong bảo mật phần mềm.

## Hướng phát triển tương lai
-   **Tăng cường dữ liệu:** Thử nghiệm với nhiều bộ dữ liệu khác nhau để cải thiện 
    khả năng tổng quát hóa và xử lý các lớp dữ liệu hiếm.
-   **Mô hình sinh (Generative Models):** Khám phá các mô hình như LLaMA và Mistral 
    để hiểu mã tốt hơn và tạo dữ liệu huấn luyện tổng hợp.
-   **Học chuyển giao (Transfer Learning):** Tùy chỉnh các chiến lược phát hiện 
    bằng cách sử dụng học chuyển giao với các mô hình sinh.

## Tài liệu tham khảo (References)

Dưới đây là các tài liệu tham khảo chính được sử dụng trong dự án:

1.  **[CodeBERT]** Z. Feng et al., "CodeBERT: A Pre-Trained Model for Programming and Natural Languages," arXiv:2002.08155, 2020.
2.  **[GraphCodeBERT]** D. Guo et al., "GraphCodeBERT: Pre-training Code Representations with Data Flow," arXiv:2009.08366, 2020.
3.  **[UniXcoder]** D. Guo et al., "UniXcoder: Unified Cross-Modal Pre-training for Code Representation," arXiv:2203.03850, 2022.
4.  **[Draper VDISC]** R. Russell et al., "Automated Vulnerability Detection in Source Code Using Deep Representation Learning," ICMLA 2018.
5.  **[EnStack]** S. Z. Ridoy et al., "EnStack: An Ensemble Stacking Framework of Large Language Models for Enhanced Vulnerability Detection in Source Code," arXiv:2411.16561v1, 2024.
