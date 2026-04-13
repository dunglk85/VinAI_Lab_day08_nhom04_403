# Group Report — Lab Day 08: RAG Pipeline

**Tên nhóm:** [Tên Nhóm]
**Thành viên:** 
1. Ngô Gia Bảo (Eval Owner)
2. Lê Kim Dũng (Tech Lead, Retrieval Owner)
3. Nguyễn Dương Linh (Documentation Owner)

---

## 1. Kết quả đạt được (Sprint 1-4)

Nhóm đã hoàn thành xây dựng hệ thống RAG trợ lý nội bộ với các thông số chính:
- **Index**: 34 chunks từ 5 tài liệu chính sách, đầy đủ metadata (source, section, effective_date, department).
- **Retrieval**: Triển khai Baseline (Dense) và Variant (Hybrid + Rerank).
- **Evaluation**: Chạy scorecard tự động cho 10 câu hỏi kiểm thử.

### Bảng so sánh kết quả:
| Metric | Baseline | Variant (Hybrid + Reranking) |
|--------|----------|-----------------------------|
| Faithfulness | 4.50/5 | 4.50/5 |
| Relevance | 4.60/5 | 4.10/5 |
| Context Recall | 5.00/5 | 5.00/5 |
| Completeness | 4.00/5 | 3.20/5 |

---

## 2. Quyết định kỹ thuật quan trọng

1. **Chiến lược Chunking**: Sử dụng kích thước 400 tokens và overlap 80 tokens. Quyết định cắt theo Heading giúp giữ được tính toàn vẹn của các điều khoản chính sách.
2. **Xử lý Alias**: Đã sửa lỗi trong bộ tiền xử lý (Preprocessor) để không bỏ lỡ thông tin tên cũ của tài liệu (ví dụ: Approval Matrix), giúp cải thiện đáng kể Context Recall.
3. **Lựa chọn Variant**: Nhóm thử nghiệm Hybrid Retrieval nhằm tối ưu việc tìm kiếm mã lỗi (ERR-code), kết hợp với LLM Reranking để lọc noise. Kết quả cho thấy bản Baseline đã rất mạnh sau khi fix indexing, và Rerank cần tuning kỹ hơn để tránh làm giảm Completeness.

---

## 3. Phân bổ công việc

| Thành viên | Nhiệm vụ chính |
|------------|----------------|
| **Ngô Gia Bảo** | Tech Lead, implement `index.py`, `rag_answer.py`, debug pipeline. |
| **[Thành viên 2]** | Eval Owner, thiết kế test questions, chạy scorecard, phân tích lỗi. |
| **[Thành viên 3]** | Documentation Owner, viết `architecture.md`, `tuning-log.md`. |

---

## 4. Tự đánh giá

Nhóm tự đánh giá hoàn thành **100% các yêu cầu** của Sprint 1 đến Sprint 4. Hệ thống hoạt động ổn định, không bịa thông tin (abstain tốt) và có cấu trúc metadata chuẩn cho việc mở rộng sau này.
