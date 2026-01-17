# Checkpoint Strategy Guide

## TL;DR - Khuyến nghị

**Cho Google Colab (khuyến nghị):**
```yaml
save_steps: 500  # Save mỗi 500 steps để phòng disconnect
```

**Cho máy local ổn định:**
```yaml
save_steps: 0    # Chỉ save cuối epoch, training nhanh hơn
```

**Cho training rất dài (>2 giờ/epoch):**
```yaml
save_steps: 200  # Save thường xuyên hơn để an toàn
```

---

## Chi tiết 2 chiến lược

### Chiến lược 1: Save mỗi N steps + cuối epoch (Mặc định)

**Config:**
```yaml
training:
  save_steps: 500  # Save every 500 steps
```

**Cách hoạt động:**
- **Mid-epoch**: Lưu `recovery_checkpoint` và `checkpoint_epochX_stepY` mỗi 500 steps
- **End-of-epoch**: Lưu `last_checkpoint` (epoch=X, step=0) và xóa `recovery_checkpoint`
- **Resume**: Ưu tiên `last_checkpoint`, fallback sang `recovery_checkpoint`

**Ưu điểm:**
- ✅ **An toàn nhất**: Không mất tiến trình dù crash bất cứ lúc nào
- ✅ **Phù hợp Colab**: Bảo vệ khỏi disconnect ngẫu nhiên
- ✅ **Phù hợp epoch dài**: Với 1+ giờ/epoch, tránh mất nhiều công

**Nhược điểm:**
- ❌ **Chậm hơn ~5-10%**: Mỗi lần save mất 10-30s
- ❌ **Tốn disk**: Mỗi checkpoint ~500MB

**Khi nào dùng:**
- Training trên Google Colab
- Epoch dài (>30 phút)
- Mạng/máy không ổn định
- Dataset lớn (mất nhiều thời gian nếu phải train lại)

---

### Chiến lược 2: Chỉ save cuối epoch

**Config:**
```yaml
training:
  save_steps: 0  # Disable mid-epoch checkpointing
```

**Cách hoạt động:**
- **Mid-epoch**: KHÔNG lưu
- **End-of-epoch**: Lưu `last_checkpoint` (epoch=X, step=0)
- **Resume**: Chỉ có `last_checkpoint`

**Ưu điểm:**
- ✅ **Nhanh nhất**: Không bị gián đoạn bởi I/O
- ✅ **Đơn giản**: Không có checkpoint trung gian gây rối
- ✅ **Tiết kiệm disk**: Chỉ 1 checkpoint/model

**Nhược điểm:**
- ❌ **Rủi ro mất tiến trình**: Crash giữa epoch → mất hết công
- ❌ **Không phù hợp Colab**: Session timeout → mất tiến trình

**Khi nào dùng:**
- Training local trên máy ổn định
- Epoch ngắn (<20 phút)
- Có UPS/backup power
- Debug/thử nghiệm nhanh

---

## So sánh hiệu suất

### Thời gian training (ước tính)

**Với dataset của bạn:**
- 1270 batches/epoch × 3.22s/batch = 68 phút/epoch

| Chiến lược | Save overhead | Thời gian/epoch | Rủi ro mất công |
|------------|---------------|-----------------|-----------------|
| `save_steps: 500` | ~60s (3 lần × 20s) | **~69 phút** | Max 27 phút |
| `save_steps: 0` | ~20s (1 lần) | **~68 phút** | Max 68 phút |

**Tiết kiệm:** ~1 phút/epoch = ~10 phút/10 epochs

**Rủi ro:** Nếu crash giữa epoch, mất tối đa 68 phút vs 27 phút

### Disk usage

Với CodeBERT (~500MB/checkpoint):

| Chiến lược | Checkpoints | Total disk |
|------------|-------------|------------|
| `save_steps: 500` | ~6 mid-epoch + 1 last | **~3.5 GB** |
| `save_steps: 0` | 1 last only | **~500 MB** |

---

## Quản lý checkpoints

### Xóa checkpoint cũ để tiết kiệm dung lượng:

```bash
# Xem danh sách checkpoints
python scripts/cleanup_checkpoints.py \
  --checkpoint_dir /content/drive/MyDrive/EnStack_Data/checkpoints/codebert

# Xóa TẤT CẢ mid-epoch checkpoints (giữ lại last_checkpoint)
python scripts/cleanup_checkpoints.py \
  --checkpoint_dir /content/drive/MyDrive/EnStack_Data/checkpoints/codebert \
  --keep-last 0 \
  --auto

# Giữ lại 2 checkpoint gần nhất
python scripts/cleanup_checkpoints.py \
  --checkpoint_dir /content/drive/MyDrive/EnStack_Data/checkpoints/codebert \
  --keep-last 2
```

### Tự động cleanup sau mỗi epoch:

Code đã tự động xóa `recovery_checkpoint` khi epoch hoàn thành. Nhưng các checkpoint `checkpoint_epochX_stepY` vẫn giữ lại.

Nếu muốn tự động xóa, có thể sửa code hoặc chạy cleanup script định kỳ.

---

## Khuyến nghị cụ thể

### Cho bài toán của bạn (EnStack trên Colab):

**Recommended:**
```yaml
save_steps: 500  # Hoặc 1000 nếu muốn nhanh hơn
```

**Lý do:**
1. ✅ Colab có thể disconnect bất ngờ → Cần mid-epoch checkpoint
2. ✅ 68 phút/epoch khá dài → Mất nhiều nếu crash
3. ✅ 3 models × 10 epochs = 30 epochs total → Rủi ro crash cao
4. ❌ Overhead ~1 phút/epoch là chấp nhận được

### Tối ưu hóa:

**Option 1: Tăng save_steps lên nếu muốn nhanh hơn**
```yaml
save_steps: 1000  # Chỉ save 1-2 lần/epoch
```
- Nhanh hơn (~30s overhead thay vì 60s)
- Vẫn an toàn hơn save_steps=0
- Rủi ro mất công tối đa ~54 phút

**Option 2: Disable cho debug nhanh**
```yaml
save_steps: 0  # Chỉ dùng khi test code
```
- Chỉ dùng khi chạy thử 1-2 epochs
- KHÔNG dùng cho training chính thức

---

## Best Practices

### 1. Luôn kiểm tra checkpoint sau khi training

```bash
python scripts/debug_checkpoint.py \
  --checkpoint_path /content/drive/MyDrive/EnStack_Data/checkpoints/codebert/last_checkpoint
```

### 2. Backup checkpoint quan trọng

```bash
# Backup checkpoint best model
cp -r /content/drive/MyDrive/EnStack_Data/checkpoints/codebert/last_checkpoint \
      /content/drive/MyDrive/EnStack_Data/backups/codebert_epoch5
```

### 3. Monitor disk usage

```bash
du -sh /content/drive/MyDrive/EnStack_Data/checkpoints/*
```

### 4. Cleanup định kỳ

Sau mỗi vài epochs, chạy cleanup để xóa mid-epoch checkpoints cũ:

```bash
python scripts/cleanup_checkpoints.py \
  --checkpoint_dir /content/drive/MyDrive/EnStack_Data/checkpoints/codebert \
  --keep-last 0 \
  --auto
```

---

## Troubleshooting

### Vấn đề: Training chậm vì save checkpoint quá nhiều

**Giải pháp:**
- Tăng `save_steps` lên (500 → 1000)
- Hoặc disable: `save_steps: 0`

### Vấn đề: Crash giữa epoch, mất nhiều tiến trình

**Giải pháp:**
- Giảm `save_steps` xuống (500 → 200)
- Đảm bảo recovery_checkpoint được lưu

### Vấn đề: Hết dung lượng Google Drive

**Giải pháp:**
- Chạy cleanup script
- Xóa best_model checkpoints cũ nếu không cần

### Vấn đề: Checkpoint bị corrupt

**Giải pháp:**
- Code mới đã có atomic save, không còn bị corrupt
- Nếu vẫn gặp, kiểm tra Google Drive sync
