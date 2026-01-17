# Checkpoint và Resume Training - Phân tích và Sửa lỗi

## Vấn đề báo cáo:
User báo rằng đã train xong epoch 1 và đã bắt đầu epoch 2 (khoảng 8/1270 batches), sau đó dừng và resume. Nhưng khi resume, checkpoint hiển thị `epoch=1, step=1000` và training chạy lại epoch 1 từ step 1000.

## Phân tích chi tiết:

### 1. Cơ chế lưu checkpoint

Code có 2 điểm lưu checkpoint:

**A. Mid-epoch checkpoint** (src/trainer.py:320-321):
```python
if (step + 1) % save_steps == 0:
    self.save_checkpoint("last_checkpoint", epoch=epoch, step=(step + 1))
```
- Lưu mỗi 500 steps (default save_steps=500)
- Epoch 1: Lưu tại step 500, 1000 (ghi đè lên file `last_checkpoint`)

**B. End-of-epoch checkpoint** (src/trainer.py:646):
```python
self.save_checkpoint("last_checkpoint", epoch=epoch, step=0)
```
- Lưu khi epoch hoàn thành
- `step=0` đánh dấu epoch đã xong

### 2. Vấn đề phát hiện được:

#### **Vấn đề #1: Checkpoint có thể bị ghi đè**
Cả mid-epoch và end-of-epoch đều lưu vào cùng file `last_checkpoint`, dẫn đến:
- Checkpoint cuối epoch CÓ THỂ bị ghi đè bởi checkpoint mid-epoch của epoch tiếp theo
- Nếu có lỗi giữa validation và save checkpoint, checkpoint cuối epoch không được lưu

#### **Vấn đề #2: Không có error handling khi save checkpoint**
```python
self.model.save_pretrained(str(save_path))  # Có thể fail
torch.save(state_dict, save_path / "training_state.pth")  # Có thể fail
```
Nếu fail, checkpoint bị corrupt hoặc không được lưu, nhưng code không báo lỗi rõ ràng.

#### **Vấn đề #3: Logging không đầy đủ**
- Không log rõ khi epoch hoàn thành
- Không log chi tiết trạng thái checkpoint khi load
- Khó debug khi có vấn đề

#### **Vấn đề #4: Progress bar gây hiểu nhầm**
```python
progress_bar = tqdm(enumerate(self.train_loader), 
                    total=total_batches, 
                    initial=resume_step)
```
- Với `initial=1000`, tqdm bắt đầu đếm từ 1000
- Nhưng vòng lặp vẫn iterate qua TẤT CẢ batches (từ 0 đến 1269)
- Khi skip 1000 batches đầu, progress bar vẫn tăng mỗi iteration
- Output "1812it" = 1000 (initial) + 812 (iterations)
- Có thể gây hiểu lầm về số batches thực sự được train

### 3. Kịch bản có thể xảy ra (giải thích checkpoint epoch=1, step=1000):

**Kịch bản A: Epoch 1 chưa hoàn thành**
1. Train epoch 1 đến step 1000
2. Lưu checkpoint (epoch=1, step=1000)
3. **Training bị dừng** (user Ctrl+C, crash, out of memory, etc.)
4. Checkpoint cuối epoch không được lưu
5. Resume → Load checkpoint (epoch=1, step=1000)

**Kịch bản B: Checkpoint cuối epoch bị fail**
1. Train epoch 1 hoàn thành (1270/1270 batches)
2. Chạy validation
3. Cố gắng lưu checkpoint (epoch=1, step=0)
4. **Save FAILED** (Google Drive sync issue, permission, disk full, etc.)
5. Checkpoint vẫn là (epoch=1, step=1000) từ lần lưu trước
6. Resume → Load checkpoint (epoch=1, step=1000)

**Kịch bản C: Checkpoint bị ghi đè**
1. Epoch 1 hoàn thành → Lưu (epoch=1, step=0) ✅
2. Epoch 2 bắt đầu
3. Epoch 2, step 500 → Lưu (epoch=2, step=500) - GHI ĐÈ file
4. User rollback/restore Google Drive về version cũ
5. Checkpoint quay về (epoch=1, step=1000)

## Các sửa đổi đã thực hiện:

### ✅ Sửa #1: Atomic checkpoint save với error handling
- Lưu vào temp directory trước
- Chỉ move sang final location khi thành công
- Tạo backup trước khi ghi đè
- Log chi tiết lỗi nếu save fail
- Không crash training nếu save fail

### ✅ Sửa #2: Enhanced logging
- Log chi tiết khi load checkpoint (epoch, step, total_batches, completion status)
- Log rõ ràng khi lưu mid-epoch vs end-of-epoch checkpoint
- Log chi tiết logic resume (epoch đã xong hay chưa)
- Thêm visual indicators (✅, ⏸️, ➡️) để dễ đọc

### ✅ Sửa #3: Lưu total_batches vào checkpoint
- Giúp xác định chính xác epoch đã hoàn thành hay chưa
- Phát hiện nếu dataset size thay đổi giữa các lần chạy

### ✅ Sửa #4: Improved resume logic
- Kiểm tra `step >= total_batches` để detect epoch hoàn thành
- Log rõ ràng % progress nếu mid-epoch
- Log số batches còn lại

## Tools hỗ trợ debug:

### scripts/debug_checkpoint.py
Phân tích chi tiết checkpoint state:
```bash
python scripts/debug_checkpoint.py --checkpoint_path /path/to/checkpoint
```

### scripts/fix_checkpoint_epoch.py
Sửa thủ công checkpoint để đánh dấu epoch đã xong:
```bash
python scripts/fix_checkpoint_epoch.py --checkpoint_path /path/to/checkpoint --epoch 1
```

## Khuyến nghị:

1. **Kiểm tra log chi tiết** của lần train trước để xác định:
   - Epoch 1 có hoàn thành không (tìm "Epoch 1 COMPLETED" hoặc validation metrics)
   - Có lỗi nào khi save checkpoint không
   
2. **Nếu chắc chắn epoch 1 đã xong:**
   - Sử dụng `scripts/fix_checkpoint_epoch.py` để fix checkpoint
   - Hoặc xóa checkpoint và train lại từ đầu

3. **Kiểm tra Google Drive sync** nếu train trên Colab:
   - Đảm bảo Drive có đủ dung lượng
   - Kiểm tra file checkpoint có bị conflict không
   - Xem Drive có message lỗi sync không

4. **Monitoring trong lần train tiếp theo:**
   - Theo dõi log "📥 Saving end-of-epoch checkpoint"
   - Kiểm tra "✅ Checkpoint saved" confirmation
   - Xác nhận checkpoint state sau mỗi epoch

## Log mẫu sau khi sửa:

```
============================================================
RESUMING TRAINING FROM CHECKPOINT
Checkpoint path: /content/drive/MyDrive/EnStack_Data/checkpoints/codebert/last_checkpoint
============================================================
============================================================
LOADED CHECKPOINT STATE:
  Epoch: 1
  Step: 1000
  Total Batches (saved): 1270
  Best Val F1: 0.0667
  Best Val Acc: 0.5000
  Status: ⏸️  Epoch 1 INCOMPLETE (78.7% done)
============================================================

Current dataset: 1270 batches/epoch

⏸️  Epoch 1 is INCOMPLETE
   Progress: 1000/1270 batches (78.7%)
   Remaining: 270 batches
➡️  Will resume WITHIN epoch 1 from step 1000
============================================================

============================================================
STARTING TRAINING: 10 epochs (from epoch 1)
============================================================

============================================================
EPOCH 1/10
  Resuming from step 1000
============================================================
```

Với các sửa đổi này, user sẽ thấy rõ ràng:
- Checkpoint hiện tại ở đâu
- Epoch đã hoàn thành hay chưa
- Training sẽ resume từ đâu
- Nếu có lỗi khi save checkpoint
