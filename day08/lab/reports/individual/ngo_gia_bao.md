# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Ngô Gia Bảo

**Vai trò trong nhóm:** Tech Lead / Retrieval Owner

**Ngày nộp:** 13/04/2026  

**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong dự án lab này, tôi đóng vai trò là Tech Lead và Retrieval Owner, chịu trách nhiệm chính trong việc xây dựng và kết nối các thành phần của pipeline RAG. Cụ thể, tôi đã thực hiện:
- **Sprint 1 & 2:** Xây dựng quy trình Indexing hoàn chỉnh trong `index.py`, bao gồm việc thiết kế logic chia nhỏ tài liệu (chunking) theo Section heading và Paragraph để giữ ngữ cảnh tự nhiên. Tôi cũng đã viết hàm `get_embedding()` sử dụng OpenAI API.
- **Debugging:** Tôi đã trực tiếp xử lý các lỗi kỹ thuật quan trọng như lỗi mismatch dimension (3072 vs 1536) trong ChromaDB bằng cách thực hiện reset collection và tối ưu hóa việc nạp biến môi trường từ file `.env`.
- **Sprint 3:** Triển khai cấu hình Hybrid Retrieval và Rerank trong `rag_answer.py` để so sánh với bản Baseline.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau buổi làm lab, khái niệm mà tôi hiểu rõ nhất chính là **tầm quan trọng của Preprocessing (Tiền xử lý)** trong Indexing. Ban đầu, tôi cho rằng kết quả tìm kiếm sai là do thuật toán Retrieval chưa đủ mạnh (Dense vs Hybrid). Tuy nhiên, qua thực tế debug câu hỏi về "Approval Matrix", tôi nhận ra rằng nếu bước Preprocessing vô tình loại bỏ thông tin quan trọng (như các dòng chú thích alias ở đầu file) thì dù thuật toán tìm kiếm có tốt đến đâu cũng không thể trả về kết quả đúng. Ngoài ra, tôi cũng hiểu sâu hơn về **Evaluation Loop** — việc sử dụng LLM-as-Judge để chấm điểm tự động giúp chúng ta nhanh chóng nhận ra các lỗi hallucination hoặc lỗi thiếu thông tin (completeness) mà nếu kiểm tra thủ công sẽ mất rất nhiều thời gian.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Khó khăn lớn nhất mà tôi gặp phải là lỗi **401 Unauthorized** và **Dimension Mismatch**. Cụ thể, khi hệ thống đã có sẵn API key cũ trong môi trường (system environment variable), hàm `load_dotenv()` thông thường không ghi đè được dữ liệu từ file `.env`, dẫn đến việc gọi API liên tục thất bại. Ngoài ra, việc thay đổi giữa các model embedding của OpenAI (từ `large` sang `small`) yêu cầu phải xóa và khởi tạo lại vector store, nếu không ChromaDB sẽ báo lỗi không tương thích kích thước vector. Điều này dạy cho tôi bài học rằng khi xây dựng pipeline, việc quản lý trạng thái của database (reset/init) và quản lý chặt chẽ biến môi trường là những bước nền tảng vô cùng quan trọng trước khi nghĩ đến việc tối ưu hóa thuật toán.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** "Approval Matrix để cấp quyền hệ thống là tài liệu nào?" (q07)

**Phân tích:**
Trong lần thử nghiệm đầu tiên, hệ thống Baseline trả lời sai hoàn toàn vì bước Preprocessing đã coi dòng *"Ghi chú: Tài liệu này trước đây có tên Approval Matrix for System Access"* là metadata rác và loại bỏ nó khỏi nội dung index. Kết quả là điểm Faithfulness chỉ đạt 2/5 và Completeness đạt rất thấp.

Sau khi tôi điều chỉnh lại logic trong hàm `preprocess_document` để giữ lại các dòng chú thích (notes) quan trọng, điểm Baseline đã cải thiện đáng kể (Faithfulness tăng lên 3/5). Tuy nhiên, khi chuyển sang **Variant (Hybrid + Rerank)**, tôi ngạc nhiên thấy kết quả không tốt hơn đáng kể so với bản Baseline đã fix indexing. Điều này cho thấy với tập dữ liệu nhỏ và cấu trúc tài liệu rõ ràng, Dense Retrieval đơn thuần đã làm rất tốt. Việc thêm Rerank thậm chí còn gây rủi ro loại bỏ nhầm các chunk chứa alias nếu điểm số reranking không đủ cao. Điều này chứng minh rằng việc tối ưu hóa "Data Quality" ở bước Indexing thường mang lại hiệu quả cao hơn và chi phí thấp hơn so với việc cố gắng làm phức tạp hóa bước Retrieval.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ tập trung vào việc **Prompt Engineering** cho bước Generation và điều chỉnh **Top-k select**. Hiện tại điểm Completeness của một số câu như `q01` vẫn chưa đạt 5/5 do model trả lời quá ngắn gọn. Tôi muốn thử tăng `top_k_select` từ 3 lên 5 và điều chỉnh prompt để ép model phải liệt kê đầy đủ các điều kiện ngoại lệ khi trả lời chính sách hoàn tiền, thay vì chỉ tóm tắt ý chính.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
