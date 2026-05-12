# PLAN: Tối ưu hóa thuật toán Build Tree cho InGPO (Level-Wise Global Batching)

## 1. Mục tiêu
- **Tăng tốc độ xây dựng cây**: Giảm thời gian inference từ O(Total Nodes) calls xuống O(Max Depth) calls bằng cách batch hóa toàn cục.
- **Duy trì tính đúng đắn**: Giữ nguyên logic Trigger, Pruning và cấu trúc cây hiện tại.
- **Hiệu suất mục tiêu**: Tương đương hoặc nhanh hơn SPO, đặc biệt khi độ sâu (depth) và số lượng mẫu (batch size) lớn.

## 2. Phân tích vấn đề hiện tại
- **Nút cổ chai**: Code hiện tại duyệt DFS/BFS từng sample độc lập. Dù có `asyncio`, việc gọi model vẫn diễn ra tuần tự theo sample hoặc theo node đơn lẻ.
- **Lãng phí tài nguyên**: GPU không được tận dụng tối đa vì mỗi lần gọi chỉ xử lý 1 prefix (hoặc rất ít), trong khi throughput của LLM cao nhất khi batch size lớn.
- **Overhead I/O**: Việc lưu/cache từng cây riêng lẻ tạo thêm độ trễ không cần thiết trong quá trình sinh.

## 3. Giải pháp kỹ thuật: Level-Wise Global Batching

Thay vì: `Loop Samples -> Loop Depth -> Expand Node`
Sẽ là: `Loop Depth -> Gather All Active Nodes (Global) -> Batch Expand -> Distribute Results`

### Các thay đổi cụ thể:

#### A. File: `ingpo/spo/inference/tree_inference_strategy.py`
- **Chức năng cũ**: Hàm `generate()` lặp qua từng `problem`, gọi `_construct_tree()` đệ quy/async cho từng cây.
- **Chức năng mới**:
    - Khởi tạo danh sách các cây (roots) cho toàn bộ batch samples cùng lúc.
    - Vòng lặp chính chạy theo `depth` (từ 0 đến `max_depth`).
    - Trong mỗi vòng lặp `depth`:
        1. **Gather**: Thu thập tất cả các node "lá" đang hoạt động (chưa bị prune/trigger stop) từ TẤT CẢ các cây trong batch.
        2. **Batch Expand**: Gọi hàm `global_batch_expand()` với danh sách node này.
        3. **Distribute**: Nhận kết quả (các node con mới) và gắn đúng vào cây cha tương ứng dựa trên `sample_id` và `node_id`.
        4. **Check Stop**: Cập nhật trạng thái trigger/prune cho vòng lặp tiếp theo.

#### B. File: `ingpo/spo/inference/expansion.py`
- **Chức năng cũ**: `EfficientIIDExpander.expand(node)` nhận 1 node, sinh ra các con cho node đó.
- **Chức năng mới**:
    - Thêm phương thức `global_batch_expand(nodes: List[Node])`.
    - **Logic bên trong**:
        1. Trích xuất `prefix` (text) từ danh sách `nodes`.
        2. Tạo một prompt batch duy nhất chứa tất cả các prefix này.
        3. Gọi model inference **một lần duy nhất** với `batch_size = len(nodes) * num_samples_per_node`.
        4. Parse output batch, tách rời kết quả theo từng node gốc.
        5. Áp dụng logic `trigger` (nếu có) lên từng kết quả trước khi trả về.
    - **Lưu ý**: Cần xử lý padding cẩn thận để đảm bảo các sequence trong cùng một batch có độ dài phù hợp (có thể cần group by length nếu chênh lệch quá lớn, nhưng ưu tiên batch thô trước).

#### C. File: `ingpo/spo/inference/node.py`
- **Bổ sung**:
    - Thêm thuộc tính `sample_idx` (chỉ số của sample trong batch tổng) vào class `Node`.
    - Đảm bảo mỗi node mang thông tin định danh để khi nhận kết quả từ batch lớn, ta biết gán nó vào cây nào.

#### D. File: `ingpo/spo/configs/...` (Tùy chọn)
- Thêm tham số `global_batch_limit`: Giới hạn số node tối đa được gom vào một lần expand (tránh OOM nếu batch quá lớn). Nếu số node vượt quá limit, sẽ chia nhỏ thành các sub-batch trong cùng một level.

## 4. Lộ trình thực hiện

### Giai đoạn 1: Chuẩn bị dữ liệu và cấu trúc (Node & Strategy)
1. Sửa class `Node` để lưu `sample_idx`.
2. Viết lại hàm `generate()` trong `tree_inference_strategy.py`:
   - Khởi tạo list `trees`.
   - Tạo vòng lặp `for depth in range(max_depth)`.
   - Implement logic `gather_active_nodes(trees)`.

### Giai đoạn 2: Implement Batch Expansion Core
1. Sửa `expansion.py`: Implement `global_batch_expand()`.
2. Đảm bảo logic gọi model (`model.generate` hoặc tương đương) nhận input là list các prefix và trả về list các kết quả.
3. Implement logic map kết quả ngược lại vào các node cha.

### Giai đoạn 3: Tích hợp Logic Trigger & Pruning
1. Đảm bảo sau khi expand batch, logic kiểm tra điều kiện dừng (trigger) được áp dụng cho từng node con.
2. Đánh dấu các node không thỏa mãn để không đưa vào vòng lặp `depth` tiếp theo.

### Giai đoạn 4: Testing & Validation
1. **Unit Test**: Kiểm tra `global_batch_expand` trả về đúng số lượng và thứ tự kết quả.
2. **Integration Test**: Chạy thử với batch nhỏ (ví dụ 4 samples), so sánh kết quả cây sinh ra với code cũ (phải trùng khớp về nội dung và cấu trúc).
3. **Benchmark**: Đo thời gian chạy với các độ sâu khác nhau (3, 5, 10) và so sánh với SPO.

## 5. Rủi ro và Phương án dự phòng
- **Rủi ro OOM (Out Of Memory)**: Khi gom quá nhiều node vào một batch, GPU có thể tràn bộ nhớ.
  - *Giải pháp*: Thêm cơ chế chia nhỏ batch (chunking) trong `global_batch_expand` nếu số node vượt ngưỡng an toàn.
- **Rủi ro Padding inefficiency**: Các prefix độ dài khác nhau gây lãng phí tính toán.
  - *Giải pháp*: Nhóm các node có độ dài tương đương vào cùng một sub-batch (Bucketing) trước khi gọi model.
- **Khó khăn với Logic Trigger phức tạp**: Nếu trigger phụ thuộc vào trạng thái toàn cục của cây.
  - *Giải pháp*: Truyền thêm metadata của cây vào context khi batch (nếu cần), hoặc chấp nhận hy sinh một chút hiệu năng để xử lý trigger riêng sau bước generate.

## 6. Tiêu chí hoàn thành (Definition of Done)
- [ ] Code chạy thành công không lỗi cú pháp/logic cơ bản.
- [ ] Kết quả sinh cây (nội dung text) trùng khớp 100% với phiên bản cũ (khi seed giống nhau).
- [ ] Thời gian chạy giảm đáng kể (ít nhất 30-50% với depth > 5).
- [ ] Tài liệu README.md được cập nhật phần "Performance Optimization".

---
*Người lập kế hoạch: AI Assistant*
*Ngày: 2024*
