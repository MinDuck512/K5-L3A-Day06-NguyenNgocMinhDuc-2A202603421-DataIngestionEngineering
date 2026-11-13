# Corpus cho bài lab

Chủ đề: Python cơ bản. Nguồn: Python Tutorial chính thức, snapshot 3.14.10, truy cập ngày 2026-11-13 theo múi giờ +07:00.

| Chương | Tài liệu | Ký tự sau chuẩn hóa | Đơn vị nguồn |
|---|---|---:|---:|
| 3 | An Informal Introduction to Python | 17.034 | 7 |
| 4 | More Control Flow Tools | 35.861 | 15 |
| 5 | Data Structures | 22.861 | 10 |
| 8 | Errors and Exceptions | 21.837 | 10 |
| Tổng | 4 tài liệu | 97.593 | 42 |

## Cách tính khối lượng

Đây là HTML, không có số trang PDF. Văn bản phần nội dung chính được chuẩn hóa rồi chia tuần tự, không overlap, ưu tiên ranh giới đoạn, với mục tiêu 2.400 ký tự và tối thiểu 1.500 ký tự mỗi đơn vị. Phần dư ngắn được gộp vào đơn vị cuối trước đó. Snapshot hiện tại có đơn vị dài từ 1.681 đến 3.485 ký tự.

Các đơn vị nguồn không phải chunk truy xuất. Ba chiến lược A/B/C sẽ chia chunk bên trong cùng các đơn vị này ở bước sau. `page` là số thứ tự đơn vị trong mỗi tài liệu, không phải số trang vật lý. Khi ghi gold, dùng đồng thời `document_id` và `page`, hoặc `unit_id` tương ứng.

## Các file

- `raw/`: HTML gốc của 4 chương và bản giấy phép đầy đủ.
- `processed/`: văn bản nội dung chính sau chuẩn hóa; giữ thụt dòng trong code.
- `manifest.json`: URL, quyền sử dụng, thời điểm truy cập, phiên bản, hash, danh sách đơn vị và ranh giới ký tự.
- `source_units.jsonl`: mỗi dòng là một đơn vị nguồn có nội dung, ID, vị trí và metadata.
- `ATTRIBUTION.md`: nguồn, giấy phép và các thay đổi đã thực hiện.

Metadata `chapter` lấy từ tiêu đề chương, có các giá trị 3/4/5/8, được chọn làm trường filter. `version` và `language` được đọc từ HTML; chúng không phân biệt các tài liệu trong snapshot này.

## Tái tạo dữ liệu

Script dùng thư viện chuẩn Python, không cần model embedding, không gọi LLM và không chạy benchmark. Từ thư mục `starter`, chạy:

```powershell
python -B -m src.ingestion --prepare-corpus
```

Lệnh tải nguồn trực tuyến và tạo snapshot/manifest khi chưa có benchmark. Nếu benchmark đã tồn tại, chương trình từ chối tải lại để giữ nguyên corpus. Sau khi đã chốt bộ câu hỏi và thí nghiệm, giữ nguyên dữ liệu đã lưu; việc tải lại có thể làm corpus thay đổi nếu website được cập nhật.

## Việc học viên cần kiểm tra ở bước 2

1. Đọc phần đầu của mỗi chương và đối chiếu bản HTML gốc với text đã chuẩn hóa.
2. Mở manifest, hiểu cách truy từ một `unit_id` về đúng file và ranh giới ký tự.
3. Điền họ tên, MSSV và lớp trong `STUDENT.md`; phần corpus đã được điền.
4. Bộ 20 câu hỏi đã được lưu trong `benchmark.json`: 15 dev ở chương 3/4/5, 5 test ở chương 8. Đọc phần policy trong file để hiểu split, tiêu chí chọn cấu hình và giới hạn. Chưa chạy retrieval.
