# Hướng dẫn Khôi phục Training sau khi Sửa lỗi Checkpoint

## Tóm tắt Vấn đề
Hệ thống lưu checkpoint trên Google Drive gặp lỗi **mất đồng bộ** do cơ chế `shutil.move()` không ổn định trên FUSE filesystem. Điều này dẫn đến các thư mục `.tmp` không được chuyển đổi thành checkpoint thực tế.

## Thay đổi đã thực hiện (Bản Tối ưu hóa)
**File**: `src/trainer.py` - Hàm `save_checkpoint()`

**Cải tiến chính**:
1. **Chiến lược "Local-First"**:
   - Khi phát hiện Google Drive, hệ thống sẽ tạo thư mục tạm trên **Local VM SSD** (`/content/temp_checkpoints`) thay vì trên Drive.
   - Việc ghi file model/optimizer (nặng hàng trăm MB) diễn ra **cực nhanh** trên SSD.
   - Tránh hoàn toàn lỗi mạng/timeout khi `save_pretrained` đang chạy.

2. **Copy An toàn & Sync**:
   - Sau khi ghi xong ở Local, thực hiện **một lệnh copy duy nhất** lên Drive.
   - Gọi `os.sync()` để ép hệ điều hành đẩy dữ liệu từ RAM xuống đĩa.
   - Tăng thời gian chờ lên 3s.

3. **Xác minh kép**: Kiểm tra file tồn tại và kích thước > 0 sau khi copy.

## Các Bước Khôi phục Training

### Bước 1: Dọn dẹp các thư mục `.tmp` rác

**Trên Google Colab**, chạy script sau trong một cell mới:

```python
!python scripts/cleanup_gdrive_checkpoints.py --dry-run
```

Xem danh sách các file sẽ bị xóa. Nếu đồng ý, chạy lại không có `--dry-run`:

```python
!python scripts/cleanup_gdrive_checkpoints.py
```

**Hoặc xóa thủ công** (nếu script gặp lỗi):

```python
import shutil
from pathlib import Path

checkpoint_dir = Path("/content/drive/MyDrive/EnStack_Data/checkpoints/codebert")

# Xóa các thư mục .tmp
for tmp_dir in checkpoint_dir.glob(".tmp_*"):
    print(f"Deleting: {tmp_dir.name}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

# Xóa các thư mục .backup
for backup_dir in checkpoint_dir.glob(".backup_*"):
    print(f"Deleting: {backup_dir.name}")
    shutil.rmtree(backup_dir, ignore_errors=True)

print("✅ Cleanup complete!")
```

### Bước 2: Xác định checkpoint hợp lệ để resume

Checkpoint duy nhất **đáng tin cậy** hiện tại là:
- **`best_model`** (Epoch 2, Step 0, F1 = 0.7806)

Các checkpoint khác (như `checkpoint_epoch3_step1000`) **có thể bị hỏng** do lỗi save nên không nên dùng.

### Bước 3: Kiểm tra code mới đã được cập nhật

**Quan trọng**: Đảm bảo file `src/trainer.py` đã được cập nhật với bản sửa lỗi.

Để kiểm tra, mở file và xem hàm `save_checkpoint()` có đoạn code này không:

```python
# Detect if we're saving to Google Drive
is_gdrive = "/content/drive/" in str(self.output_dir)
```

Nếu có -> Code đã được cập nhật ✅  
Nếu không -> Pull lại code mới nhất từ repository

### Bước 4: Khởi động lại Training

Chạy script training **từ đầu** (hoặc resume từ `best_model`):

```python
# Trong Google Colab
!python scripts/train.py --resume_from /content/drive/MyDrive/EnStack_Data/checkpoints/codebert/best_model
```

### Bước 5: Theo dõi và Xác minh

**Quan sát Log**:
- Khi lưu checkpoint, bạn sẽ thấy dòng:
  ```
  Google Drive detected - using COPY method for safety
  ```
- Checkpoint sẽ xuất hiện **ngay lập tức** sau khi log báo `✅ Checkpoint saved`

**Kiểm tra thủ công** (trong một cell riêng):
```python
import os
from pathlib import Path

checkpoint_dir = Path("/content/drive/MyDrive/EnStack_Data/checkpoints/codebert")
print("Checkpoints hiện tại:")
for item in sorted(checkpoint_dir.iterdir()):
    if item.is_dir():
        print(f"  📁 {item.name}")
```

## Lưu ý Quan trọng

### ⚠️ KHÔNG dùng checkpoint đã bị hỏng
Các checkpoint được tạo **trước khi sửa lỗi** (như `checkpoint_epoch3_step1000` nếu tồn tại) có thể không đầy đủ. Hãy bắt đầu lại từ `best_model` (Epoch 2).

### ⚠️ Chờ đủ thời gian
Sau khi log báo checkpoint đã lưu, hãy chờ **ít nhất 3 giây** trước khi kiểm tra thủ công trong Drive UI (web interface Drive rất chậm).

### ⚠️ Không tắt Colab giữa chừng
Google Drive sync có thể mất thời gian. Nếu bạn ngắt kết nối Colab ngay sau khi checkpoint được tạo, file có thể không được flush xuống Drive kịp.

## Cách Test nhanh (Optional)

Trước khi chạy training đầy đủ, bạn có thể test cơ chế lưu checkpoint:

```python
!python scripts/test_checkpoint_save.py
```

Nếu thấy `✅ ALL TESTS PASSED`, nghĩa là cơ chế lưu đã hoạt động đúng.

## Troubleshooting

### Vấn đề: Vẫn thấy thư mục `.tmp` sau khi training
**Nguyên nhân**: Google Drive bị lag, file đã bị xóa nhưng Drive chưa cập nhật UI.

**Giải pháp**: 
1. Refresh trang Drive (F5)
2. Kiểm tra bằng lệnh `ls` trong Colab thay vì nhìn trên giao diện web

### Vấn đề: Checkpoint không xuất hiện sau khi log báo "saved"
**Nguyên nhân**: Lỗi permission hoặc Drive đầy dung lượng.

**Giải pháp**:
1. Kiểm tra dung lượng Drive: `!df -h /content/drive/`
2. Kiểm tra quyền ghi: `!touch /content/drive/MyDrive/test.txt && rm /content/drive/MyDrive/test.txt`

### Vấn đề: Training bị crash khi save checkpoint
**Nguyên nhân**: OOM (Out of Memory) do model quá lớn.

**Giải pháp**:
- Tắt tính năng lưu mid-epoch checkpoint: `save_steps=0` trong config
- Chỉ giữ lại checkpoint cuối epoch (`last_checkpoint`) và best model (`best_model`)

## Tóm tắt

✅ **Code đã được sửa** - Cơ chế lưu đã tương thích với Google Drive  
✅ **Script dọn dẹp** - `scripts/cleanup_gdrive_checkpoints.py`  
✅ **Checkpoint an toàn** - Resume từ `best_model` (Epoch 2)  
✅ **Hướng dẫn đầy đủ** - Tài liệu này

**Next Steps**: Chạy lại training và theo dõi log để đảm bảo checkpoint được lưu thành công.
