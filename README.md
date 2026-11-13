# Day 06 · Data Ingestion Engineering

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

Trong bài làm hiện tại, corpus và benchmark nằm trong `starter/data/`; mã nguồn nằm trong `starter/src/`, kết quả thí nghiệm sẽ nằm trong `starter/results/`. Xem [lệnh kiểm tra benchmark](starter/README.md#4-corpus-và-benchmark-của-bài-làm-hiện-tại). Đây là các vị trí bài làm tương ứng với checklist nộp bài.

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.

## 4. Cách chạy bài làm hiện tại — bước embedding

Đã hoàn thiện corpus HTML, benchmark, chia chunk, embedding thật, Chroma lưu bền vững và metric dev/test. Các bảng nhãn hỗ trợ bởi Codex đã có để tác giả rà soát; chưa ghi nhận tác giả đã chấm/xác nhận. Mã dùng các module sẵn có trong `starter/src/`; kết quả embedding chỉ bổ sung một file `starter/results/embeddings.npz`.

Các lệnh sau chạy từ `starter/`, dùng môi trường ảo đặt ở cấp thư mục ngoài để tránh đường dẫn dài trên Windows:

```powershell
# Kiểm tra corpus và bộ câu hỏi
..\..\.venv\Scripts\python.exe -B -m src.run --validate-benchmark

# Chia chunk; chỉ chạy khi cần tạo lại artifact này
..\..\.venv\Scripts\python.exe -B -m src.run --build-chunks

# Lần đầu: tải model và tạo embedding local
..\..\.venv\Scripts\python.exe -u -B -m src.run --embed-chunks

# Các lần sau: dùng model đã có trong .venv/minilm
..\..\.venv\Scripts\python.exe -u -B -m src.run --embed-chunks --offline

# Kiểm tra file vector, không cần nạp model hoặc chạy lại embedding
..\..\.venv\Scripts\python.exe -B -m src.run --validate-embeddings
```

Dependencies của môi trường đang chạy được ghi trong `starter/requirements.txt`. Model được tải vào `.venv/minilm` của Python đang dùng, ở ngoài root repo; không đưa môi trường ảo/model cache lên Git. Lần tải đầu khoảng 470 MB cho weights, cộng tokenizer/config. Code chỉ tải các file cần cho PyTorch, không tải các bản ONNX, OpenVINO hoặc TensorFlow.

### Model và cách xử lý

- Provider: Sentence Transformers chạy local bằng CPU, 4 thread trong lần chạy hiện tại.
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Revision cố định: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
- Vector: 384 chiều, float32, mean pooling theo cấu hình model, chuẩn hóa L2.
- Document: chỉ text của chunk; query: chỉ text câu hỏi. Cùng model, không thêm prefix, không đưa gold hoặc filter vào text embedding.
- Chi phí API trả phí: 0 USD; việc tạo vector thực hiện trên máy local.

Theo [model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), giới hạn xử lý mặc định là 128 token. [Cấu hình kiến trúc](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/blob/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/config.json) có 512 vị trí. Bài làm chủ động đặt `max_seq_length=512` cho cùng model ở tất cả cấu hình, trước khi chạy retrieval. Đây là thay đổi giới hạn xử lý, không phải huấn luyện thêm model; chất lượng vẫn phải được đo trên dev.

Tokenizer đếm toàn bộ đầu vào, gồm special token, với `truncation=False`. Nếu đầu vào vượt 512 token, chương trình báo lỗi trước khi encode. Kết quả kiểm tra thực tế:

| Bộ đầu vào | Số text/vector | Token dài nhất | Số text vượt mặc định 128 | Số text vượt cấu hình 512 |
|---|---:|---:|---:|---:|
| A / fixed | 235 | 220 | 136 | 0 |
| B / paragraph | 265 | 227 | 59 | 0 |
| C | Dùng lại toàn bộ B | Như B | Như B | 0 |
| Câu hỏi dev | 15 | 35 | 0 | 0 |

### File kết quả và kiểm tra

`embeddings.npz` chứa hai ma trận document `(235,384)` và `(265,384)`, một ma trận dev query `(15,384)`, ID theo đúng thứ tự hàng, số token của mỗi text và metadata JSON. C tham chiếu bộ paragraph của B, không lưu thêm bản vector trùng. Năm câu test chưa được encode hoặc truy xuất ở bước này.

Metadata ghi revision, hash weights thực tế, model/provider, phiên bản Python/thư viện, cấu hình token, cách xử lý query/document, thời gian chuẩn bị model, thời gian encode riêng từng bộ và chi phí API. Có hash của corpus, benchmark và file chunk để phát hiện đầu vào đã đổi. Thời gian nạp/tải model được tách khỏi thời gian encode; chúng chưa phải thời gian ingest đầy đủ vào vector store.

Đã kiểm tra ma trận đủ hàng/cột, đúng float32, không có NaN/Inf, vector có độ dài L2 gần 1, thứ tự ID khớp chunk/query, token không vượt giới hạn và xử lý một query riêng khớp xử lý theo batch. Một đầu vào giả lập quá dài chỉ nằm trong RAM được dùng để xác nhận chương trình từ chối cắt ngầm. File được mở lại bằng `allow_pickle=False` để kiểm chứng kết quả đã lưu. Chế độ dùng model offline cũng đã được chạy kiểm tra.

Các file đầu ra được cập nhật tại cùng đường dẫn khi chạy lại; không sinh bản sao theo timestamp. Sau khi tạo lại chunk hoặc sửa benchmark, vector cũ sẽ bị báo không khớp hash và cần được tạo lại trước khi sử dụng. Giữ nguyên test khi điều chỉnh các cấu hình trên dev.

File NumPy này là artifact trung gian để tái sử dụng vector; phần Chroma được mô tả ở mục 5 và kết quả benchmark ở mục 6. Nhật ký đã có Hit@1/Hit@3 thực tế; các nhãn rubric được ghi rõ là hỗ trợ bởi Codex, chờ tác giả rà soát.

## 5. Bước 6 — Chroma lưu bền vững và truy xuất dev

Từ `starter/`, chạy hai lệnh thành hai tiến trình Python riêng:

```powershell
# Đồng bộ database, kiểm tra nạp lại, dọn chunk cũ và lưu top-3 dev
..\..\.venv\Scripts\python.exe -u -B -m src.run --ingest-store

# Mở lại database trong tiến trình mới, đối chiếu mọi bản ghi
..\..\.venv\Scripts\python.exe -u -B -m src.run --check-store
```

Hai lệnh dùng vector đã có trong `embeddings.npz`, không nạp model hoặc gọi embedding API. Kỳ vọng `Store ingestion OK` và `Store persistence OK`. Cờ kiểm tra yêu cầu có database và report trước đó; không tự tạo database rỗng rồi báo thành công.

### Database và cấu hình

`src/store.py` hiện dùng `ChromaVectorStore`, thay phần khung PostgreSQL bằng phương án Chroma đã chọn. Backend là Chroma **1.5.9**, `PersistentClient`, vector **384 chiều**. [PersistentClient](https://docs.trychroma.com/reference/python/client) lưu database trong `starter/results/chroma/`.

| Cấu hình | Collection | Số bản ghi |
|---|---|---:|
| A | day6_fixed | 235 |
| B | day6_paragraph | 265 |
| C | day6_paragraph, dùng chung B | 265 |

Tổng vật lý là 500 bản ghi. Mỗi bản ghi có chunk ID, text, vector, `document_id`, `source`, `page` và metadata đã kiểm chứng, gồm trường `chapter` kiểu số nguyên.

Index dùng [HNSW approximate nearest-neighbor search](https://docs.trychroma.com/docs/collections/configure), metric cosine. Các giá trị được backend trả lại và kiểm tra: `space=cosine`, `ef_construction=100`, `ef_search=100`, `max_neighbors=16`, `resize_factor=1.2`, `sync_threshold=100`. Collection còn lưu fingerprint của model/revision/weights/số chiều/cấu hình xử lý để từ chối trộn vector khác model. `embedding_function=None`; chương trình đưa vector đã tính vào `query_embeddings` và không để Chroma tự chọn model embedding khác.

Chroma tự quản lý các file SQLite/index trong thư mục database. Thư mục này được thêm vào `.gitignore` vì có thể tái tạo bằng lệnh ingest. `retrieval.json` và các artifact nhỏ vẫn được giữ làm minh chứng.

### Nạp lại và tài liệu thay đổi

`upsert()` ở adapter nhận **toàn bộ snapshot của một bộ chunk**, kiểm tra ID/vector/metadata, upsert theo batch, rồi xóa ID cũ không còn trong snapshot. Nếu nội dung hoặc cách chunk thay đổi, ID mới được nạp và các chunk cũ được loại khỏi collection. Không truyền một batch tài liệu riêng lẻ vào hàm đồng bộ toàn snapshot. Snapshot rỗng bị từ chối để tránh làm mất collection khi đọc dữ liệu lỗi.

Kiểm tra thực tế đã chạy:

- Lượt nạp ban đầu: A từ 0 lên 235, B từ 0 lên 265.
- Lượt nạp lại cùng snapshot: A giữ 235, B giữ 265, không thêm ID mới.
- Mô phỏng snapshot thay đổi: tạm bỏ một chunk thật của A, database giảm từ 235 xuống 234 và xóa đúng ID cũ; sau đó khôi phục snapshot đầy đủ về 235. Corpus gốc không đổi.
- Đối chiếu toàn bộ 500 bản ghi: ID, text, metadata và vector khớp artifact đầu vào.
- Đóng client, kết thúc tiến trình ghi, mở lại bằng tiến trình Python khác: đọc đúng cả 500 bản ghi và top-3 dev giữ cùng ID trong lần kiểm tra này.

Các số 0 ban đầu là của lần tạo database đầu tiên; các lần chạy lại sẽ bắt đầu từ số bản ghi đang có. Lệnh luôn ghi kết quả đo thực tế của lần chạy hiện tại.

### Top-3 và filter

Đã chạy 15 câu dev trên A/B/C, tương đương 45 lượt truy xuất, lưu đủ 135 kết quả. Trace nằm trong **`starter/results/retrieval.json`**, gồm câu hỏi, ID, text, cosine distance, score, metadata, filter yêu cầu/filter đã dùng và thời gian query. Score được định nghĩa là `1 - cosine_distance`, cao hơn nghĩa là gần hơn theo cosine; đây chưa phải xác suất đúng hoặc Hit@1/Hit@3.

A/B không filter. C chuyển `chapter` được nêu trước trong câu hỏi sang [`where`](https://docs.trychroma.com/docs/querying-collections/metadata-filtering) của Chroma. q07 lọc chương 4, q12 lọc chương 5: cả hai trường hợp đều trả top-3 đúng chương. Mười ba câu dev không có filter cho cùng top-3 B/C trong lần chạy này. Không dùng gold source/trang/đáp án để tạo filter.

Kiểm tra bổ sung đã xác nhận từ chối vector sai chiều/vector zero, filter theo gold source, filter gắn nhầm cho A/B, snapshot rỗng và collection có fingerprint model khác. Một filter thử không tồn tại trả danh sách rỗng, không tự bỏ filter để lấy kết quả ngoài phạm vi. Các trường hợp thử không được thêm vào benchmark.

`retrieval.json` còn ghi phiên bản/cấu hình backend, fingerprint đầu vào, số bản ghi trước/sau nạp, kiểm tra dọn chunk cũ, thời gian upsert, kích thước database, kết quả mở lại ở tiến trình mới và ghi chú lỗi setup đã sửa. Cả 4 chương vẫn nằm trong kho truy xuất; chỉ 5 câu hỏi test chưa chạy. Các kiểm tra metadata ở bước này là đối chiếu tự động, chưa thay audit thủ công trên test.

Quy trình chọn trên dev rồi đánh giá test đã được thực hiện và mô tả ở mục 6. Các kết quả lưu trữ ở mục này là kiểm tra chức năng; nhãn rubric và khả năng trả lời có bằng chứng được trình bày riêng.

## 6. Bước 7 — benchmark dev/test và nhãn đánh giá hỗ trợ

Kết quả được bổ sung ngay trong [starter/results/retrieval.json](starter/results/retrieval.json), không tạo thêm file kết quả. `src/evaluation.py` tính chỉ số và tổng hợp các nhãn đã ghi; `src/run.py` điều phối chốt cấu hình, chạy test và kiểm tra tính nhất quán.

### Định nghĩa metric và bằng chứng tương đương

Một hit bắt buộc có một chunk ở nguồn/đơn vị được chấp nhận và chứa đủ bằng chứng cho câu hỏi. Hit@1 xét rank 1; Evidence Hit@3 xét có ít nhất một chunk đủ trong top-3. Hợp nhiều chunk được ghi như chẩn đoán, không thay định nghĩa chỉ số chính. Kết quả lưu cả tử/mẫu và tỷ lệ.

Ngoài khớp đầy đủ gold sau chuẩn hóa Unicode/khoảng trắng, hai trường hợp dev của A được Codex đối chiếu với nguồn và ghi bằng chứng tương đương trong `evidence_reviews`: q07 có mệnh đề else của loop chạy khi không có break; q11 giữ đủ hai câu về pop phần tử cuối và IndexError khi list rỗng dù mất phần mở đầu gold. Các nhãn này dùng chung theo question/chunk cho mọi cấu hình; có quote, nguồn, lý do và tên người hỗ trợ. Tác giả cần rà soát trước khi nhận là nhãn của mình.

Để minh bạch ảnh hưởng của việc đối chiếu, baseline chỉ khớp nguyên gold vẫn được lưu: A dev Hit@1=4/15, Hit@3=7/15; B/C dev Hit@1=5/15, Hit@3=8/15. Số liệu chính dưới đây gồm hai bằng chứng tương đương đã ghi rõ; không điều chỉnh câu hỏi/corpus/model/chiến lược để nâng điểm sau khi xem test.

| Cấu hình | Tập | Hit@1 | Evidence Hit@3 |
|---|---|---|---|
| A | dev | 6/15 (40,0%) | 8/15 (53,3%) |
| B | dev | 5/15 (33,3%) | 8/15 (53,3%) |
| C | dev | 5/15 (33,3%) | 8/15 (53,3%) |
| A | test | 2/5 (40,0%) | 3/5 (60,0%) |
| B | test | 3/5 (60,0%) | 4/5 (80,0%) |
| C | test | 3/5 (60,0%) | 4/5 (80,0%) |

### Chọn và cố định trước khi chạy test

A được chọn theo thứ tự tiêu chí đã chốt: Hit@3 dev bằng nhau, A có Hit@1 dev cao hơn. Không cần dùng các tiêu chí phụ về thời gian/số chunk. Quyết định, bảng ứng viên, cấu hình và hash đầu vào được lưu trước khi encode và truy xuất câu hỏi test: freeze lúc `2026-11-13T19:13:19.791584+07:00`, bắt đầu test lúc `2026-11-13T19:13:19.794261+07:00`.

Sau đó mới encode năm câu test bằng cùng model local đã cache và truy xuất A/B/C. Năm vector query test được lưu trong chính `retrieval.json`; file embedding dev/document không bị thay đổi. Test B/C tốt hơn A ở một câu nhưng lựa chọn A vẫn giữ nguyên. Không chọn lại cấu hình dựa vào test; test chỉ có năm câu nên một câu tương đương 20 điểm phần trăm và chưa đại diện toàn corpus.

### Các bảng nhãn — cần tác giả rà soát

**Các điểm dưới đây là nhãn hỗ trợ bởi Codex, chưa phải xác nhận tác giả đã chấm thủ công.** Tất cả có reviewer, lý do, ID chunk/trace và `student_review=pending`. Phương pháp dùng AI và nguy cơ thiên lệch được ghi trong từng nhóm audit; không phải chấm mù hoàn toàn.

| Cấu hình | Coherence /2 (10 vị trí) | Metadata đủ/đúng trên test | Tuân thủ filter test | Grounding: số câu mức 0/1/2 | Grounding TB /2 |
|---|---:|---:|---:|---|---:|
| A | 1,0 | 15/15 | 1/1 | 1 / 1 / 3 | 1,4 |
| B | 1,9 | 15/15 | 1/1 | 1 / 0 / 4 | 1,6 |
| C | 1,9 | 15/15 | 1/1 | 1 / 0 / 4 | 1,6 |

- **Coherence:** đọc 20 chunk khác nhau ở cùng 10 vị trí nguồn đã chọn trước, rồi ghi 30 nhãn A/B/C. B/C cùng text nên cùng nhãn. Điểm 0 là không hiểu do mất ý/chủ thể; 1 là hiểu một phần, thiếu ngữ cảnh; 2 là tự hiểu được và giữ trọn ý chính. Có lý do riêng từng mẫu; mẫu B ở phần pattern còn tham chiếu ví dụ “above” nên chỉ đạt 1. Không suy điểm từ Hit@3.
- **Metadata:** kiểm tra toàn bộ 45 bản ghi top-3 test với raw HTML: URL, heading/chương, version, language, ID đơn vị và ranh giới ký tự. Nhãn là kiểm tra nguồn tự động kết hợp xem xét của Codex, chờ tác giả xác nhận. `source/page/offset` giúp truy vết; `chapter` giúp lọc. Metadata đúng không chứng minh evidence đã đủ.
- **Grounding:** đọc top-3 của từng câu test cho từng cấu hình; điểm 0 không có bằng chứng cho các ý được hỏi, 1 có một phần nhưng thiếu điều kiện/ngữ cảnh, 2 đủ để trả lời và truy nguồn. Đây là độ đủ của evidence retrieval, không phải độ chính xác câu trả lời LLM. q18 đạt 0 cả ba; q20 của A có thông tin một phần về exception group nhưng thiếu định nghĩa gói danh sách, nên đạt 1.
- **Filter:** một câu test có yêu cầu chương là q18. Cả ba tình cờ trả toàn chương 8, nên đều 1/1. Trên dev, A/B chỉ 1/2 câu tuân thủ đầy đủ phạm vi, C là 2/2; tuy nhiên q12 vẫn không có bằng chứng deque trong top-3 của B/C. Trường version/language không thay đổi trong corpus này nên ít giúp phân biệt nguồn.

Để rà soát: `audits.coherence.labels` liên kết đến chunk trong `chunking.json`; `audits.grounding.labels` liên kết đến `test_traces`; `audits.metadata.labels` có từng phép đối chiếu nguồn. Hai nhãn đủ bằng chứng tương đương nằm trong `evidence_reviews`. Không đổi trường pending thành đã xác nhận khi chưa thực sự xem xét.

### Tác động chiến lược, lỗi và giới hạn

A–B cho thấy đánh đổi: B có chunk mạch lạc hơn ở mẫu đánh giá nhưng không thắng Hit@1 dev; A có overlap nên tổng text/token cần embed cao hơn dù số chunk ít hơn. A–B thay cả ranh giới và overlap nên không cô lập hai ảnh hưởng. Thời gian chunking, embedding và vector upsert được giữ riêng trong `resource_comparison`; đây là số đo thành phần từ các lần chạy cụ thể, không phải phép đo end-to-end chuẩn hóa. Chi phí API trả phí là 0 USD.

B–C cùng index và dữ liệu nên khác biệt đến từ filter. Bộ câu hiện tại chưa cho thấy tăng Hit@1/Hit@3 nhờ C; có lợi ích giới hạn phạm vi trên dev nhưng câu test filter vốn đã đúng chương ở mọi cấu hình. Không kết luận filter vô ích hoặc chiến lược nào tốt nhất tổng quát.

Hai ví dụ lỗi có thể mở trace để kiểm chứng:

1. **q01/dev:** gold `python314-introduction:unit-001` nêu phép chia `/` trả float, nhưng top-3 thiên về list/dictionary/set hoặc số hỗn hợp. Cả ba cấu hình thiếu bằng chứng đúng.
2. **q18/test:** gold `python314-errors:unit-004` nêu vị trí và điều kiện chạy else của try/except. Top-3 lại là các đoạn về except/finally/ExceptionGroup. Đúng chương vẫn chưa đủ nội dung để trả lời.

Giới hạn khác: câu hỏi tiếng Việt/nguồn tiếng Anh, model đa ngữ không bảo đảm mọi truy vấn; max sequence length đã tăng từ 128 lên 512 nhưng chưa có thí nghiệm riêng; nhãn AI có nguy cơ thiên lệch và cần tác giả rà soát. Không chỉnh tham số bằng test rồi tiếp tục gọi đó là holdout độc lập.

### Lệnh chấm và kiểm tra

Từ `starter/`:

```powershell
# Chấm lại trace dev đã lưu, không chạy test
..\..\.venv\Scripts\python.exe -B -m src.run --score-dev

# Lần đầu cố định quyết định trước khi chạy test; khi đã có test thì chỉ trả lại kết quả đã lưu
..\..\.venv\Scripts\python.exe -u -B -m src.run --run-test

# Tổng hợp metric/nhãn đã ghi và cập nhật experiment_log.csv, không truy xuất lại
..\..\.venv\Scripts\python.exe -B -m src.run --score-results

# Kiểm tra 60 trace, 180 kết quả, công thức metric, thứ tự freeze/test và ID nguồn của các nhãn
..\..\.venv\Scripts\python.exe -B -m src.run --validate-results
```

Kỳ vọng `Benchmark result validation OK`, đồng thời vẫn có `human_author_review_pending=true`. Các kiểm tra này xác minh tính nhất quán của kết quả đã lưu, không biến nhãn AI thành nhãn đã được tác giả xác nhận. Nếu CSV đang mở trong Excel, lưu và đóng file để lệnh tổng hợp cập nhật được nhật ký.

Sau khi đã freeze, chạy lại `--ingest-store` với cùng đầu vào giữ nguyên quyết định, trace test, metric và audit; chỉ cập nhật kiểm tra nạp lại ở trường riêng. Đầu vào/tham số khác bị từ chối trong cùng thí nghiệm đã chốt. Đã chạy kiểm tra bảo toàn các dữ liệu frozen khi nạp lại và mở database ở tiến trình mới.
