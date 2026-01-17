# Hướng dẫn Khắc phục Sự cố (Troubleshooting Guide)

Tài liệu này cung cấp giải pháp cho các vấn đề thường gặp khi triển khai và chạy dự án EnStack.

## Mục lục
- [Vấn đề về Môi trường](#vấn-đề-về-môi-trường)
- [Vấn đề về Dữ liệu](#vấn-đề-về-dữ-liệu)
- [Vấn đề về Huấn luyện](#vấn-đề-về-huấn-luyện)
- [Vấn đề về Git và GitHub](#vấn-đề-về-git-và-github)
- [Vấn đề về Google Colab](#vấn-đề-về-google-colab)

---

## Vấn đề về Môi trường

### Lỗi: ModuleNotFoundError

**Triệu chứng:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Nguyên nhân:** Thư viện chưa được cài đặt hoặc môi trường Python không đúng.

**Giải pháp:**
```bash
# Cài đặt lại tất cả dependencies
pip install -r requirements.txt

# Hoặc cài đặt từng thư viện cụ thể
pip install transformers==4.20.0
pip install torch>=1.10.0
```

### Lỗi: ImportError khi import src modules

**Triệu chứng:**
```python
ImportError: cannot import name 'EnStackModel' from 'src.models'
```

**Giải pháp:**
```python
# Thêm đường dẫn src vào PYTHONPATH
import sys
sys.path.insert(0, '/path/to/EnStack_Reproduction/src')

# Hoặc trong Colab
sys.path.append('/content/EnStack_Reproduction/src')
```

### Lỗi: CUDA out of memory

**Triệu chứng:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Giải pháp:**
1. **Giảm batch size:**
   ```yaml
   # Trong configs/config.yaml
   training:
     batch_size: 8  # Thay vì 16
   ```

2. **Sử dụng gradient accumulation:**
   ```python
   # Trong trainer.py
   accumulation_steps = 2
   ```

3. **Xóa cache CUDA:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

## Vấn đề về Dữ liệu

### Lỗi: File not found khi load dữ liệu

**Triệu chứng:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/train.pkl'
```

**Giải pháp:**
1. Kiểm tra đường dẫn trong `config.yaml`:
   ```yaml
   data:
     root_dir: "/content/drive/MyDrive/EnStack_Data"
   ```

2. Verify file tồn tại trên Google Drive:
   ```python
   import os
   print(os.listdir('/content/drive/MyDrive/EnStack_Data'))
   ```

3. Mount lại Google Drive nếu cần:
   ```python
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```

### Lỗi: Dữ liệu bị corrupt

**Triệu chứng:**
```
pickle.UnpicklingError: invalid load key
```

**Giải pháp:**
1. Download lại file dữ liệu từ nguồn gốc
2. Kiểm tra quá trình upload lên Drive có bị gián đoạn không
3. Sử dụng checksum để verify integrity:
   ```bash
   md5sum train_processed.pkl
   ```

---

## Vấn đề về Huấn luyện

### Lỗi: Loss không giảm hoặc tăng

**Triệu chứng:** Training loss không giảm sau nhiều epochs.

**Giải pháp:**
1. **Kiểm tra learning rate:**
   ```yaml
   training:
     learning_rate: 2e-5  # Thử giảm xuống 1e-5
   ```

2. **Kiểm tra data loading:**
   ```python
   # Verify labels có đúng không
   for batch in train_loader:
       print(batch['labels'])
       break
   ```

3. **Thêm gradient clipping:**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

### Lỗi: Validation accuracy thấp bất thường

**Nguyên nhân:** Model có thể bị overfit hoặc data leakage.

**Giải pháp:**
1. Kiểm tra train/val split có chồng chéo không
2. Thêm regularization (dropout, weight decay)
3. Sử dụng early stopping:
   ```python
   if val_loss > best_val_loss:
       patience_counter += 1
       if patience_counter > 3:
           break
   ```

### Lỗi: NaN loss

**Triệu chứng:**
```
Loss: nan
```

**Giải pháp:**
1. Giảm learning rate
2. Kiểm tra input data có giá trị NaN không:
   ```python
   import numpy as np
   if np.isnan(batch['input_ids'].numpy()).any():
       print("Found NaN in input!")
   ```
3. Thêm gradient clipping
4. Kiểm tra label encoding có đúng không

---

## Vấn đề về Git và GitHub

### Lỗi: Git pull fails

**Triệu chứng:**
```
error: Your local changes to the following files would be overwritten by merge
```

**Giải pháp:**
```bash
# Option 1: Stash changes
git stash
git pull origin main
git stash pop

# Option 2: Hard reset (CẢNH BÁO: mất tất cả thay đổi local)
git fetch origin
git reset --hard origin/main
```

### Lỗi: Permission denied (publickey)

**Triệu chứng:**
```
Permission denied (publickey).
fatal: Could not read from remote repository.
```

**Giải pháp:**
```bash
# Sử dụng HTTPS thay vì SSH
git remote set-url origin https://github.com/username/repo.git

# Hoặc thiết lập SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"
```

### Lỗi: Large file không push được

**Triệu chứng:**
```
remote: error: File is XXX MB; this exceeds GitHub's file size limit of 100 MB
```

**Giải pháp:**
1. Thêm vào `.gitignore`:
   ```
   *.pkl
   *.pth
   *.bin
   data/
   ```

2. Sử dụng Git LFS cho file lớn:
   ```bash
   git lfs install
   git lfs track "*.pkl"
   ```

---

## Vấn đề về Google Colab

### Lỗi: Session timeout

**Triệu chứng:** Colab ngắt kết nối sau vài giờ.

**Giải pháp:**
1. Sử dụng Colab Pro để có session dài hơn
2. Lưu checkpoint thường xuyên:
   ```python
   # Sau mỗi epoch
   torch.save({
       'epoch': epoch,
       'model_state': model.state_dict(),
       'optimizer_state': optimizer.state_dict()
   }, f'/content/drive/MyDrive/checkpoints/epoch_{epoch}.pth')
   ```

3. Kích hoạt script chống timeout:
   ```javascript
   // Paste vào Console của browser
   function ClickConnect(){
       console.log("Working"); 
       document.querySelector("colab-toolbar-button#connect").click()
   }
   setInterval(ClickConnect,60000)
   ```

### Lỗi: Runtime disconnected

**Giải pháp:**
1. Restart runtime: Runtime → Restart runtime
2. Clear output: Edit → Clear all outputs
3. Kiểm tra RAM/GPU usage:
   ```python
   !nvidia-smi
   !free -h
   ```

### Lỗi: Drive mount fails

**Triệu chứng:**
```
Timeout waiting for notebook to connect to Drive
```

**Giải pháp:**
```python
# Force remount
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Hoặc restart runtime và mount lại
```

---

## Các Vấn đề Khác

### Performance chậm khi training

**Giải pháp:**
1. Kiểm tra có đang sử dụng GPU không:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")
   ```

2. Tăng số workers cho DataLoader:
   ```python
   train_loader = DataLoader(
       dataset, 
       batch_size=16, 
       num_workers=2  # Thêm dòng này
   )
   ```

3. Sử dụng mixed precision training:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

### Model không load được checkpoint

**Triệu chứng:**
```
RuntimeError: Error(s) in loading state_dict
```

**Giải pháp:**
```python
# Load với strict=False
model.load_state_dict(checkpoint['model_state'], strict=False)

# Hoặc kiểm tra keys
checkpoint_keys = set(checkpoint['model_state'].keys())
model_keys = set(model.state_dict().keys())
print("Missing:", model_keys - checkpoint_keys)
print("Extra:", checkpoint_keys - model_keys)
```

---

## Liên hệ Hỗ trợ

Nếu vấn đề vẫn chưa được giải quyết:
1. Kiểm tra [Issues trên GitHub](https://github.com/your-repo/issues)
2. Mở issue mới với thông tin chi tiết:
   - Mô tả lỗi
   - Error traceback đầy đủ
   - Môi trường (Python version, CUDA version, etc.)
   - Các bước tái hiện lỗi

---

*Tài liệu được cập nhật lần cuối: 2026-01-16*
