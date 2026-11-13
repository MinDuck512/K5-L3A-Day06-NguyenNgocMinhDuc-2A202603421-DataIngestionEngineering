# Báo cáo cá nhân — Data Ingestion Engineering

**Họ tên:** Nguyễn Ngọc Minh Đức · **MSSV:** 2A202603421 · **Lớp:** L3A  
**Ngày chạy thí nghiệm:** 13/11/2026 (UTC+07:00)

## 1. Dữ liệu và cách xử lý

Tôi xây dựng kho truy xuất kiến thức Python từ bốn chương của Python Tutorial: nhập môn, điều khiển luồng, cấu trúc dữ liệu và lỗi/ngoại lệ. Tôi sử dụng snapshot **3.14.10**, có 97.593 ký tự nội dung. Tài liệu theo PSF License Version 2; ví dụ code có điều khoản bổ sung 0BSD. Nguồn, ngày tải, giấy phép và ranh giới nội dung được lưu trong [manifest.json](../../starter/data/manifest.json) và [ATTRIBUTION.md](../../starter/data/ATTRIBUTION.md).

Tôi chia nội dung thành **42 đơn vị nguồn HTML**, mỗi đơn vị ít nhất 1.500 ký tự sau chuẩn hóa, đáp ứng quy tắc của đề. Đây là đơn vị truy vết tương đương, không phải trang PDF. Chương 3/4/5/8 lần lượt có 7/15/10/10 đơn vị.

Pipeline của tôi gồm: **đọc HTML → chuẩn hóa → chia chunk → tạo embedding → lưu Chroma → truy xuất top-3 → đánh giá**. Phần chuẩn hóa bỏ menu, footer và script; chuẩn hóa Unicode, xuống dòng và khoảng trắng, giữ thụt dòng của code. Tôi giữ HTML gốc, text chuẩn hóa và offset để truy ngược về nguồn. Nguồn là HTML nên không cần OCR; chương trình báo lỗi nếu không xác minh được metadata cần thiết.

Mỗi chunk có `document_id`, `source`, `page`, `unit_id` và offset. `source` giúp mở tài liệu, `page` là số đơn vị trong tài liệu, offset chỉ đúng đoạn text. Trường `chapter` là số nguyên lấy từ heading, dùng để lọc theo chương; `version` và `language` là chuỗi lấy từ HTML. Hai trường sau giống nhau trong corpus này nên ít giúp phân biệt kết quả.

## 2. Model và nơi lưu vector

Tôi dùng **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** chạy local trên CPU, tạo vector 384 chiều và chuẩn hóa L2. Model được cố định ở revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Môi trường chạy gồm Python 3.14.7, Sentence Transformers 6.3.0 và Chroma 1.5.9; các thư viện được ghi trong [requirements.txt](../../starter/requirements.txt).

Query và chunk dùng cùng model, không thêm đáp án hoặc gold vào đầu vào. Tôi đặt giới hạn 512 token cho cả ba cấu hình, cao hơn mặc định 128 của model. Chunk dài nhất là 227 token; chương trình kiểm tra độ dài trước khi encode để tránh cắt ngầm. Ảnh hưởng chất lượng của việc tăng giới hạn chưa được đánh giá riêng.

Tôi lưu vector bằng **Chroma PersistentClient**, tìm kiếm cosine với HNSW (tìm kiếm gần đúng), lấy ba kết quả mỗi câu hỏi. Cấu hình là `ef_construction=100`, `ef_search=100`, `max_neighbors=16`, `resize_factor=1.2`, `sync_threshold=100`. Database nằm tại `starter/results/chroma/`.

A dùng collection `day6_fixed` có 235 bản ghi; B/C dùng chung `day6_paragraph` có 265 bản ghi. Tổng lưu thực tế là **500 vector**. Kiểm tra nạp lại giữ nguyên số lượng; thử bỏ một chunk rồi khôi phục cho thấy chunk cũ được xóa đúng. Khi mở bằng tiến trình Python mới, 500 bản ghi vẫn khớp text, metadata và vector đã lưu.

Lỗi setup đã ghi nhận là Chroma không trả hai trường `num_threads/batch_size` để kiểm tra; code đã sửa sang đối chiếu các trường backend thực sự cung cấp. Chi tiết ở `setup_notes` trong file kết quả. Với phương án local này, **chi phí API trả phí là 0 USD**; tôi chưa tính điện hoặc phần cứng. Docker trong starter phục vụ phương án PostgreSQL/pgvector tùy chọn.

## 3. Ba cấu hình và bộ câu hỏi

| Cấu hình | Cách chia chunk | Overlap | Filter | Số chunk |
|---|---|---:|---|---:|
| A | Cố định 500 ký tự | 50 ký tự | Không | 235 |
| B | Theo đoạn, tối đa 500 ký tự; tách đoạn dài khi cần | 0 | Không | 265 |
| C | Giống B | 0 | Theo chương nêu trong câu hỏi | 265 |

Tôi giữ nguyên corpus, model, vector store và câu hỏi giữa ba cấu hình. So sánh B–C chỉ thay filter; A–B thay cả cách chia và overlap nên chưa tách riêng được hai ảnh hưởng. Giả thuyết là overlap giúp giữ ngữ cảnh ở biên, chia theo đoạn giúp giữ trọn ý và filter giảm nhiễu ngoài chương yêu cầu.

[benchmark.json](../../starter/data/benchmark.json) có **20 câu**, chia trước thành **15 dev** (q01–q15, chương 3/4/5) và **5 test** (q16–q20, chương 8). Mỗi câu có đáp án, nguồn, đơn vị và đoạn bằng chứng. Ba câu có filter là q07: chương 4, q12: chương 5 và q18/test: chương 8. Giá trị filter lấy từ câu hỏi, không lấy từ đáp án hay nguồn gold. Câu không yêu cầu filter được C chạy không filter.

Tôi dùng Hit@1 để đo chunk đầu tiên có đủ bằng chứng, và Evidence Hit@3 để đo có ít nhất một chunk đủ bằng chứng trong top-3. Đúng nguồn/đơn vị nhưng thiếu nội dung trả lời vẫn không tính là hit; ghép nhiều chunk thiếu ý không thay cho chỉ số chính.

Tiêu chí chọn đã khai báo là: **Hit@3 dev cao nhất → Hit@1 dev cao nhất → thời gian thấp hơn → ít chunk hơn**. Thời gian ở tiêu chí phụ là thời gian nạp vector. Quyết định chọn A được lưu trước khi encode/truy xuất test vào 19:13:19 ngày 13/11/2026. Kết quả test không được dùng để chọn lại cấu hình.

## 4. Kết quả

### 4.1. Độ chính xác truy xuất

| Cấu hình | Dev Hit@1 | Dev Hit@3 | Test Hit@1 | Test Hit@3 |
|---|---|---|---|---|
| A | 6/15 (40,0%) | 8/15 (53,3%) | 2/5 (40,0%) | 3/5 (60,0%) |
| B | 5/15 (33,3%) | 8/15 (53,3%) | 3/5 (60,0%) | 4/5 (80,0%) |
| C | 5/15 (33,3%) | 8/15 (53,3%) | 3/5 (60,0%) | 4/5 (80,0%) |

Bảng này gồm **hai nhãn bằng chứng tương đương do Codex đề xuất**, tôi còn cần rà soát: q07/A giữ quy tắc `else` của vòng lặp chạy khi không có `break`; q11/A giữ đủ ý `pop()` lấy phần tử cuối và gây `IndexError` khi list rỗng. Nếu chỉ khớp đầy đủ đoạn gold, A dev đạt Hit@1=4/15 và Hit@3=7/15; B/C vẫn là 5/15 và 8/15. Vì vậy, lựa chọn A phụ thuộc vào việc chấp nhận hai nhãn này.

### 4.2. Độ mạch lạc, metadata và grounding

Coherence đo chunk có giữ trọn ý; grounding đo top-3 có đủ bằng chứng trả lời, không phải chất lượng câu trả lời LLM. Thang điểm là 0: không đủ để hiểu/trả lời, 1: được một phần, 2: đủ ý và ngữ cảnh.

| Cấu hình | Coherence TB /2 (10 mẫu) | Metadata đủ/đúng | Filter test tuân thủ | Grounding: số câu 0/1/2; TB /2 |
|---|---:|---:|---:|---|
| A | 1,0 | 15/15 | 1/1 | 1/1/3; TB 1,4 |
| B | 1,9 | 15/15 | 1/1 | 1/0/4; TB 1,6 |
| C | 1,9 | 15/15 | 1/1 | 1/0/4; TB 1,6 |

Đây là **nhãn AI hỗ trợ, tôi chưa xác nhận chấm thủ công**. Coherence dùng cùng 10 vị trí chọn trước: sắp đơn vị theo tài liệu/trang, lấy các vị trí cách đều từ đầu đến cuối danh sách rồi chọn gần trung điểm. Khi overlap, lấy chunk bắt đầu sớm nhất bao phủ vị trí. Grounding dùng cả năm câu test cho từng cấu hình. Metadata được đối chiếu trên 45 bản ghi top-3 test với HTML/text nguồn. Mẫu, ID, điểm và lý do chi tiết được giữ trong `audits` của [retrieval.json](../../starter/results/retrieval.json); điểm không được suy ra từ Hit@3 và việc chấm không hoàn toàn độc lập với thông tin kết quả.

Trên dev, A/B trả đúng phạm vi ở 1/2 câu filter, C là 2/2. Test chỉ có q18 yêu cầu filter; cả ba tình cờ trả đúng chương nên đều 1/1. Điều này chưa cho thấy filter cải thiện độ chính xác tìm bằng chứng.

### 4.3. Tài nguyên và thời gian

| Cấu hình | Tổng ký tự chunk | Tổng token | Chia chunk (s) | Embedding (s) | Nạp vector (s) |
|---|---:|---:|---:|---:|---:|
| A | 107.166 | 29.901 | 0,001584 | 7,032079 | 0,196159 |
| B | 97.110 | 27.125 | 0,002635 | 6,558771 | 0,207753 |
| C | Dùng chung B | Dùng chung B | Dùng chung B | Dùng chung B | Dùng chung B |

Đây là thời gian từng công đoạn được ghi riêng, chưa phải phép đo toàn pipeline trong một lần chạy. Token gồm cả special token. Độ dài chunk A là 52–500 ký tự, trung bình 456,03; B/C là 8–499, trung bình 366,45. C dùng chung B nên không phát sinh thêm một lần embedding hoặc lưu vector.

## 5. Phân tích và kết luận

Từ A–B, tôi thấy B giữ ý tốt hơn trong mẫu coherence, còn A có dev Hit@1 cao hơn theo nhãn hiện tại. A có ít chunk hơn nhưng overlap làm tăng tổng ký tự và token cần xử lý. Từ B–C, filter giúp giới hạn đúng chương trên dev nhưng chưa tăng Hit@1/Hit@3. Đúng chương vẫn có thể lấy sai đoạn bên trong chương.

Hai lỗi tiêu biểu của cả ba cấu hình là:

- **q01/dev:** hỏi phép chia `/` trả kiểu gì. Bằng chứng nằm ở `python314-introduction:unit-001`, nhưng top-3 không có đoạn đủ trả lời. Khác biệt ngôn ngữ và ký hiệu là giả thuyết nguyên nhân, chưa được kiểm nghiệm.
- **q18/test:** hỏi vị trí và điều kiện chạy `else` trong `try/except`. Gold ở `python314-errors:unit-004`; các kết quả lại thiếu quy tắc này. A có một chunk cùng đơn vị gold nhưng thuộc đoạn khác, cho thấy đúng nguồn/trang chưa đủ.

Theo tiêu chí dev đã khai báo, lựa chọn được lưu là **A**, còn phụ thuộc vào hai nhãn tương đương cần rà soát. B/C tốt hơn trên test một câu nhưng tôi giữ nguyên lựa chọn trước test. Test chỉ có năm câu thuộc chương 8; một câu tương đương 20 điểm phần trăm nên tôi chưa kết luận cấu hình nào tốt nhất tổng quát.

Các giới hạn khác là câu hỏi tiếng Việt/nguồn tiếng Anh, 10 mẫu coherence còn ít, nhãn có AI hỗ trợ và thời gian chưa đo lặp nhiều lần. Nếu mở rộng, tôi sẽ kiểm chứng lại gold/nhãn, thử giữ heading cùng đoạn hoặc kết hợp tìm kiếm từ khóa trên dev, rồi đánh giá bằng test mới. Các cải tiến này chưa được chạy.

## 6. Minh chứng, cách kiểm tra và việc còn lại

Tôi dùng Codex hỗ trợ code, soạn câu hỏi, phân tích và viết báo cáo. Code đã kiểm tra hash, quote/offset, vector, lưu bền vững và tính nhất quán của metric. Tôi vẫn cần tự rà soát gold, hai bằng chứng tương đương và các nhãn đánh giá; trạng thái hiện tại là `pending`.

Minh chứng gồm [chunking.json](../../starter/results/chunking.json), [embeddings.npz](../../starter/results/embeddings.npz), [retrieval.json](../../starter/results/retrieval.json) và [experiment_log.csv](../../artifacts/experiment_log.csv). Có đủ 60 lượt truy xuất và 180 kết quả top-3. Chi tiết setup/các giai đoạn chạy nằm trong [README.md](../../README.md).

Để kiểm tra kết quả đã lưu trên máy hiện tại, tôi chạy:

```powershell
Set-Location -LiteralPath 'D:\VinAI\K5-L3A-Day06-NguyenNgocMinhDuc-2A202603421-DataIngestionEngineering-main\K5-L3A-Day06-NguyenNgocMinhDuc-2A202603421-DataIngestionEngineering-main\starter'
& '..\..\.venv\Scripts\python.exe' -B -m src.run --validate-results
```

Lệnh kiểm tra đã trả `Benchmark result validation OK`, kèm `human_author_review_pending=true`. Tôi còn cần hoàn tất rà soát, điền repo/commit trong `STUDENT.md` và nộp link theo hướng dẫn lớp.
