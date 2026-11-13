# Mẫu báo cáo cá nhân — Data Ingestion Engineering

**Họ tên:** [Họ tên] · **MSSV:** [MSSV] · **Lớp:** [Lớp]  
**Ngày chạy:** [Ngày và múi giờ]

## 1. Dữ liệu và pipeline

Tôi chọn [chủ đề, nguồn, giấy phép và snapshot]. Corpus có [số trang/đơn vị hợp lệ]; tôi tính số lượng theo [quy tắc]. Manifest và dữ liệu nằm tại [đường dẫn].

Tôi xử lý dữ liệu bằng [các bước đọc, chuẩn hóa, chia chunk, embedding và lưu trữ]. Lỗi/giới hạn đã gặp là [mô tả và cách xử lý].

Tôi dùng [provider, model, revision, số chiều, phiên bản thư viện] và [vector store, phiên bản, cosine, exact/approximate search]. Query/document được xử lý bằng [cách xử lý]. Metadata gồm [trường, kiểu, nguồn giá trị và mục đích truy vết/lọc].

## 2. Cấu hình và bộ câu hỏi

| Cấu hình | Cách chia/kích thước | Overlap | Metadata/filter | Giả thuyết |
|---|---|---|---|---|
| A | [Điền] | [Điền] | [Điền] | [Điền] |
| B | [Điền] | [Điền] | [Điền] | [Điền] |
| C | [Điền] | [Điền] | [Điền] | [Điền] |

Tôi giữ nguyên [corpus/model/store/benchmark]; B–C chỉ thay [một yếu tố]. Bộ câu hỏi có [tổng số], chia trước thành [dev/test]. Các câu filter là [ID, giá trị và nguồn input]. Tôi kiểm chứng gold bằng [phương pháp, người rà soát và trạng thái thực tế].

Tiêu chí tôi khai báo trước test là [thứ tự tiêu chí]. Quyết định được lưu [thời điểm, cấu hình và đường dẫn minh chứng].

## 3. Kết quả thực tế

Hit@1 xét chunk đầu tiên đủ bằng chứng; Evidence Hit@3 xét ít nhất một chunk đủ bằng chứng trong top-3. Tôi ghi riêng bằng chứng tương đương và cách xác nhận nếu có.

| Cấu hình | Tập | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Số chunk | Thời gian và cách đo | Chi phí |
|---|---|---|---|---|---|---|
| [A/B/C] | [dev/test] | [Điền] | [Điền] | [Điền] | [Điền] | [Điền] |

| Cấu hình | Coherence TB /2 (n) | Metadata đủ/đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding số câu 0/1/2 và TB /2 |
|---|---|---|---|---|
| [A/B/C] | [Điền] | [Điền] | [Điền] | [Điền] |

Mẫu, nhãn và lý do nằm tại [đường dẫn]. Tôi dùng cùng 10 vị trí coherence và 5 câu test/cấu hình để đánh giá grounding; metadata được đối chiếu trên top-3 test. Người chấm/phương pháp là [điền đúng thực tế]. Grounding đo độ đủ của bằng chứng truy xuất. Số liệu chưa chạy hoặc nhãn chưa xác nhận được ghi rõ [trạng thái].

## 4. Phân tích và kết luận

Từ A–B, tôi nhận thấy [kết quả và đánh đổi]. Từ B–C, tôi nhận thấy [tác động của yếu tố thay đổi]. Số chunk, phân bố độ dài, tổng ký tự/token và tài nguyên là [số liệu hoặc đường dẫn].

Hai lỗi tiêu biểu là [ID câu, nguồn gold, kết quả thực tế và phân tích] và [ví dụ thứ hai]. Theo tiêu chí dev, lựa chọn là [cấu hình và lý do]; tôi ghi rõ ảnh hưởng của phương pháp chấm nếu có. Giới hạn gồm [dữ liệu, nhãn, test nhỏ, thời gian/chi phí]. Nếu mở rộng, tôi sẽ [hướng cải tiến chưa chạy].

## 5. Tái lập và sử dụng AI

Tôi chạy bằng [lệnh setup/run và thư mục làm việc]. Code, manifest, benchmark, log và trace nằm tại [đường dẫn]. Tôi dùng [công cụ AI hoặc không dùng] để [công việc], kiểm tra bằng [phương pháp]. Phần tôi chưa tự rà soát/xác nhận là [liệt kê đúng trạng thái].
