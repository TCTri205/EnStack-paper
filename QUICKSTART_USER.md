# 🎓 EnStack - Hướng Dẫn Nhanh Cho Người Mới

## 🚀 Bắt Đầu Trong 5 Phút

### Bước 1: Mở Link Này
👉 **[Nhấn vào đây để mở Colab](https://colab.research.google.com/github/TCTri205/EnStack-paper/blob/main/notebooks/EnStack_Colab_Deployment.ipynb)**

### Bước 2: Bật GPU
1. Nhìn lên góc trên cùng → **Runtime** → **Change runtime type**
2. Chọn **Hardware accelerator**: **T4 GPU**
3. Nhấn **Save**

### Bước 3: Chạy
- Nhấn **Runtime** → **Run all** (hoặc Ctrl+F9)
- Đợi khoảng 30 phút
- Xem kết quả!

---

## ❓ Câu Hỏi Thường Gặp

### Q1: Tôi cần cài gì không?
**Không!** Mọi thứ đã tự động. Bạn chỉ cần:
- Tài khoản Google (miễn phí)
- Trình duyệt web

### Q2: Tốn tiền không?
**Không!** Google Colab miễn phí (có giới hạn thời gian ~12h/session).

### Q3: Chạy ở đâu?
**Trên Google Cloud**, không phải máy tính của bạn. Máy yếu vẫn chạy được.

### Q4: Kết quả lưu ở đâu?
**Google Drive** của bạn, thư mục `EnStack_Data/checkpoints/`.

### Q5: Làm sao thay đổi tham số?
Ở **Cell 7** trong notebook, điều chỉnh:
- `EPOCHS`: Số vòng lặp (2 = nhanh, 10 = chậm nhưng chính xác hơn)
- `BATCH_SIZE`: 16 (mặc định)

### Q6: Lỗi "Training loader is not provided"?
Chạy lại **Cell 5** (Download data). Chờ đến khi thấy "✅ Data preparation complete".

### Q7: Training quá chậm?
Kiểm tra **Cell 3**. Nếu không thấy "✅ GPU detected", quay lại Bước 2.

---

## 📊 Hiểu Output

Sau khi training xong, bạn sẽ thấy:

```
FINAL RESULTS SUMMARY
====================================
Validation Metrics:
  Accuracy: 0.7850
  F1: 0.7798
  Precision: 0.7785
  Recall: 0.7851

Test Metrics:
  Accuracy: 0.8236
  F1: 0.8228
  Precision: 0.8285
  Recall: 0.8236
```

**Giải thích**:
- **Accuracy**: Tỷ lệ dự đoán đúng (càng cao càng tốt, max = 1.0)
- **F1-Score**: Cân bằng giữa precision và recall
- **Precision**: Trong những cái dự đoán "có lỗi", bao nhiêu % thật sự có lỗi
- **Recall**: Trong những cái thật sự "có lỗi", bao nhiêu % được tìm ra

---

## 🎯 Mục Tiêu Dự Án (Nói Đơn Giản)

**Input**: Đoạn code C/C++  
**Output**: Loại lỗ hổng bảo mật (0-4)

Ví dụ:
```c
void unsafe_function() {
    char buf[10];
    gets(buf);  // Lỗi: Buffer overflow!
}
```
→ Model dự đoán: **Label 1 (CWE-120: Buffer Overflow)**

---

## 📱 Liên Hệ Nhanh

- **Lỗi kỹ thuật**: Mở issue tại https://github.com/TCTri205/EnStack-paper/issues
- **Đọc chi tiết**: Xem file `HANDOVER.md`
- **Hướng dẫn đầy đủ**: Xem file `README.md`

---

**Chúc bạn thành công!** 🎉
