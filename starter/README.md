# Starter · PostgreSQL + pgvector

Bộ khởi động **tùy chọn** cho lab cá nhân. Docker Compose cung cấp phương án PostgreSQL + pgvector local của đề. Bài làm hiện đã hoàn thiện chuẩn bị corpus HTML, xác minh benchmark và chia chunk; các phần embedding, vector store và truy xuất còn TODO. Phương án dự kiến của bài làm là embedding local + Chroma, được phép theo [GUIDE.md](../GUIDE.md). Các lệnh PostgreSQL bên dưới là hướng dẫn hạ tầng gốc; xem mục 4–5 để chạy những phần đã hoàn thiện.

## Cấu trúc

```text
starter/
├── docker-compose.yml     # PostgreSQL 16 + pgvector, chỉ mở cổng local
├── init.sql               # bật extension khi tạo database mới
├── requirements.txt       # psycopg, pypdf và python-dotenv
├── .env.example           # cấu hình mẫu; copy thành .env
├── scripts/check_setup.py # kiểm tra kết nối và extension (đã chạy được)
├── src/
│   ├── models.py          # cấu trúc dữ liệu mẫu
│   ├── ingestion.py       # đọc / chuẩn hóa HTML, nạp đơn vị nguồn
│   ├── chunking.py        # chia fixed-size và theo đoạn, kiểm tra truy vết
│   ├── embeddings.py      # TODO: chọn model API hoặc local
│   ├── store.py           # TODO: lưu vector, metadata và filter
│   ├── evaluation.py      # TODO: chấm truy xuất, lưu trace
│   └── run.py             # TODO: kết nối các bước thành pipeline
├── data/                  # bạn tự chuẩn bị corpus được phép sử dụng
└── results/               # lưu kết quả của bạn
```

## 1. Cài và kiểm tra database

Cần Docker Desktop hoặc Docker Engine + Compose, Python **3.11–3.14**, và mạng để tải image/dependencies lần đầu. Các lệnh sau chạy từ thư mục `starter/`:

```bash
cd starter
python3 -m venv .venv
```

Kích hoạt `.venv`: macOS/Linux dùng `source .venv/bin/activate`; Windows PowerShell dùng `.venv\Scripts\Activate.ps1`. Sau đó:

```bash
python -m pip install -r requirements.txt
# macOS/Linux: cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Kỳ vọng: hiện tên database, phiên bản PostgreSQL và pgvector, sau đó `Setup OK`. Đây **chỉ là kiểm tra hạ tầng**, không phải kết quả benchmark. Docker dùng cổng `127.0.0.1:55433` và mật khẩu cố định **chỉ dùng local cho lab**; nếu cổng bận, sửa đồng thời port trong `docker-compose.yml` và `DATABASE_URL` trong `.env`. `init.sql` chỉ chạy khi volume mới được tạo; nếu database sẵn có chưa bật extension, xem log Docker và kiểm tra cấu hình thay vì xóa volume chứa dữ liệu.

Dừng container nhưng giữ dữ liệu:

```bash
docker compose -p day6_student_starter stop
```

## 2. Chọn embedding model

Điền key tương ứng trong `.env` nếu dùng OpenRouter, OpenAI hoặc Gemini; **không commit `.env`**. Model local không cần API key. `requirements.txt` chỉ có dependencies cơ bản: tự cài SDK hoặc thư viện local phù hợp với model bạn chọn và ghi lại version/lệnh cài trong báo cáo. Ghi tên model và số chiều; dùng cùng model cho toàn bộ chiến lược và vector query tương thích vector document.

## 3. Hoàn thiện pipeline

1. Đặt corpus vào `data/` hoặc cung cấp manifest/hướng dẫn truy cập. Đọc [GUIDE.md](../GUIDE.md) về **30 trang nội dung**, quyền dữ liệu và metadata.
2. Hoàn thiện từng hàm TODO trong `src/`. Lưu ID tài liệu, nguồn, trang/vị trí và ít nhất một trường filter có ích.
3. Xây ít nhất ba cấu hình; ví dụ A fixed-size, B theo đoạn, C như B nhưng có filter. Cần cùng corpus, model và benchmark; nạp lại không nhân đôi và xử lý chunk cũ khi dữ liệu đổi.
4. Tự viết ít nhất 20 câu hỏi kèm evidence, tách dev/test trước khi thử theo [BENCHMARK.md](../BENCHMARK.md). Lưu trace top-3 và báo cáo đủ năm tiêu chí.
5. Sau khi hoàn thiện, chạy `python -m src.run` từ thư mục `starter/`. Trước khi làm TODO, lệnh này **chủ động báo `NotImplementedError`**.

Lưu code, cấu hình, kết quả và lệnh chạy để người chấm tái lập; không commit key, dữ liệu mật hoặc dữ liệu không có quyền chia sẻ.

## 4. Corpus và benchmark của bài làm hiện tại

Bài làm giữ cấu trúc `starter/` ở trên. Corpus, manifest và benchmark đều đặt trong `starter/data/`; đầu ra thí nghiệm sẽ đặt trong `starter/results/`. Không cần tạo thêm script chuẩn bị dữ liệu: mã này nằm trong `src/ingestion.py`. File mẫu `data/format-example.json` ở root là tài liệu gốc của đề, không thuộc corpus thật.

Corpus gồm bốn chương Python Tutorial, snapshot 3.14.10, tổng 97.593 ký tự nội dung chính và 42 đơn vị nguồn HTML, mỗi đơn vị ít nhất 1.500 ký tự. Xem [manifest](data/manifest.json), [hướng dẫn dữ liệu](data/README.md) và [ghi nguồn/giấy phép](data/ATTRIBUTION.md). Các đường dẫn nội bộ trong manifest được tính từ root repo.

[Bộ benchmark](data/benchmark.json) chứa 20 câu tiếng Việt, đáp án và evidence tiếng Anh lấy nguyên văn từ corpus. Mỗi câu có document ID, số đơn vị nguồn, URL, ranh giới evidence và filter khi cần. `gold_page` là số đơn vị nguồn trong tài liệu HTML, không phải trang PDF.

| Tập | Câu hỏi | Nguồn gold | Câu có filter |
|---|---|---|---|
| Dev | q01–q15 (15 câu) | Chương 3, 4, 5 | q07: chapter=4; q12: chapter=5 |
| Test | q16–q20 (5 câu) | Chương 8 | q18: chapter=8 |

Chia theo tài liệu trước khi chạy retrieval để tránh dùng cùng đoạn gold cho dev và test. Test chỉ phủ chương 8 nên không đại diện toàn bộ corpus; từ khóa và khái niệm Python vẫn có thể giao nhau giữa các chương. Chương 8 vẫn nằm trong kho truy xuất, nhưng không dùng kết quả truy xuất của năm câu test để điều chỉnh cấu hình.

Filter chỉ lấy từ chương đã nêu tường minh trong câu hỏi, không suy ra từ đáp án hay gold source. A/B không filter; C áp dụng filter khi có và chạy không filter khi giá trị là `null`. Giữ nguyên câu hỏi/gold cho cả ba cấu hình. Nguồn bằng chứng tương đương cần được ghi nhận và đánh giá nhất quán, không loại chỉ vì khác chuỗi trích dẫn ban đầu.

Tiêu chí chọn cấu hình được chốt trong `policy` của benchmark, theo thứ tự: Evidence Hit@3 trên dev, Hit@1 trên dev, thời gian ingest thấp hơn, rồi số chunk thấp hơn. Chỉ chạy test sau khi đã chọn trên dev và cố định các cấu hình.

Từ thư mục `starter/`, chạy kiểm tra offline bằng môi trường ảo đặt ở cấp thư mục ngoài để tránh đường dẫn dài trên Windows:

```powershell
..\..\.venv\Scripts\python.exe -B -m src.run --validate-benchmark
```

Kỳ vọng `Benchmark validation OK`, 20 câu, 15 dev / 5 test, 3 câu filter, 1 câu test filter và 20 evidence được xác minh. Lệnh kiểm tra hash corpus/giấy phép, ranh giới nguồn, split, filter và evidence; chỉ in kết quả ra terminal, không tạo file phụ. Đây là kiểm tra dữ liệu benchmark, chưa phải kết quả truy xuất hoặc nhãn audit thủ công.

Codex hỗ trợ soạn câu hỏi và đối chiếu evidence. Học viên cần đọc, hiểu và rà soát đáp án; trạng thái rà soát của học viên đang là `pending` trong benchmark. Không ghi trạng thái đã rà soát khi chưa thực hiện.

Để tái tạo corpus khi chưa có benchmark, dùng `python -B -m src.ingestion --prepare-corpus`. Lệnh này từ chối tải lại nếu `benchmark.json` đã tồn tại, nhằm giữ nguyên snapshot. Embedding, vector store và retrieval vẫn là bước thực hiện tiếp theo; chạy pipeline không có cờ chọn bước hiện vẫn báo TODO.

## 5. Bước 4 — chia chunk và cấu hình A/B/C

Chạy từ `starter/`:

```powershell
..\..\.venv\Scripts\python.exe -B -m src.run --build-chunks
```

Lệnh xác minh corpus/benchmark đã cố định, nạp 42 đơn vị nguồn và tạo một file kết quả cần thiết: [results/chunking.json](results/chunking.json). File này chứa cấu hình, thống kê độ dài, thời gian riêng bước chia chunk, toàn bộ chunk và 10 vị trí dành cho audit coherence sau này. Chạy lại cập nhật cùng file; không sinh thêm bản sao kết quả. `-B` tránh tạo `__pycache__`.

| Cấu hình | Cách chia | Giới hạn | Overlap | Filter khi truy xuất | Số chunk đã tạo |
|---|---|---:|---:|---|---:|
| A | Cửa sổ cố định | 500 ký tự | 50 ký tự | Không | 235 |
| B | Gom đoạn | 500 ký tự | 0 | Không | 265 |
| C | Dùng nguyên chunk của B | 500 ký tự | 0 | Theo chương trong câu hỏi | 265, dùng chung B |

Đơn vị là ký tự Unicode, chưa phải token. 500/50 là tham số khởi đầu, chưa phải tham số tốt nhất. Các cấu hình được khai báo trong `STRATEGIES` ở `src/run.py`, đồng thời lưu snapshot trong file kết quả. Chưa dùng kết quả test để chọn tham số.

### A — cửa sổ cố định

`fixed_size()` lấy text theo các khoảng `[0,500)`, `[450,950)`, `[900,1400)`... Bước dịch là `500 - 50 = 450`. Phần cuối có thể ngắn hơn 500. Overlap lặp lại một phần ngữ cảnh ở biên, đồng thời tăng text cần embedding. Cách cắt không xét câu/đoạn nên có thể dừng giữa từ, câu hoặc code.

### B — theo đoạn

`by_paragraph()` nhận biết đoạn qua dòng trống, gom các đoạn liên tiếp nếu tổng chiều dài không vượt 500. Đoạn quá dài được chia tiếp, ưu tiên xuống dòng, rồi cuối câu, rồi khoảng trắng; chuỗi quá dài không có điểm ngắt vẫn được cắt để giữ giới hạn. Code giữ nguyên thụt dòng. Những phần dư ngắn vẫn có thể xuất hiện; không tự suy ra chunk theo đoạn luôn mạch lạc.

### C — giữ nguyên dữ liệu của B

C tham chiếu cùng bộ chunk B, có cùng ID, text và metadata. Tại bước retrieval sau này, C mới áp dụng `chapter` do câu hỏi cung cấp; câu không có filter thì C chạy không filter. Một file chỉ lưu hai bộ chunk vật lý để tránh lặp lại toàn bộ nội dung của B cho C.

B–C chỉ khác filter. A–B thay cả cách đặt ranh giới và overlap, vì vậy không thể tách riêng ảnh hưởng overlap khỏi ảnh hưởng gom đoạn khi phân tích A–B.

### Truy nguồn và kiểm tra

`load_source_units()` chuyển từng dòng dữ liệu đã cố định thành `DocumentPage`, không tải lại website hoặc đổi nội dung. Mỗi chunk giữ `document_id`, `source`, số đơn vị nguồn trong `page`, `chapter`, `version`, `language` và `unit_id`.

`unit_char_start/end` là ranh giới trong đơn vị nguồn; `document_char_start/end` là ranh giới trong văn bản chương đã chuẩn hóa. Chỉ số bắt đầu từ 0, đầu cuối không bao gồm ký tự tại `end`. ID chunk được băm từ tài liệu, vị trí, cách chia, tham số và nội dung; cùng đầu vào cho cùng ID. Đổi nội dung tạo ID mới; bước vector store sau này vẫn phải xử lý chunk cũ và kiểm tra nạp lại không nhân đôi.

Đã kiểm tra từng chunk khớp đoạn nguồn, giữ metadata, không vượt giới hạn, không mất ký tự không phải khoảng trắng và tạo lại cho cùng ID/text/metadata. Kiểm tra thêm các trường hợp: tham số sai, input rỗng, overlap, đoạn dài, chuỗi không có khoảng trắng, Unicode, thụt dòng code và 150 input sinh theo seed cố định. Các input kiểm tra chỉ ở RAM, không thay corpus hoặc tạo file test phụ.

Kết quả lần chạy đầu: A có độ dài trung bình 456,03 ký tự, tổng 107.166 ký tự; B/C trung bình 366,45, tổng 97.110 ký tự. A tăng lượng text do overlap. B bỏ một số dòng trống tại ranh giới chunk; đã xác minh không mất ký tự nội dung. 42 đơn vị nguồn chứa 97.516 ký tự; con số 97.593 trong manifest còn tính thêm khoảng trắng giữa các đơn vị khi chúng nằm trong văn bản chương đầy đủ.

Mười vị trí coherence được chọn theo quy tắc cố định từ nguồn, trước khi chấm điểm: sắp xếp theo document ID/page, lấy đều từ đầu đến cuối, chọn ký tự không phải khoảng trắng gần giữa mỗi đơn vị. Với chunk overlap, lấy chunk bao phủ có vị trí bắt đầu sớm nhất. A/B/C dùng cùng vị trí; chưa gán nhãn hoặc điểm coherence.

[Nhật ký thí nghiệm](../artifacts/experiment_log.csv) đã có tham số và giả thuyết A/B/C. Các ô Hit@1/Hit@3 còn trống vì chưa chạy retrieval. Thời gian trong file chunking chỉ đo thuật toán chia, không phải thời gian ingest đầy đủ. Số token và việc kiểm tra giới hạn tokenizer sẽ thực hiện khi kết nối embedding model; chưa có chỉ số embedding, vector store hoặc độ chính xác truy xuất.
