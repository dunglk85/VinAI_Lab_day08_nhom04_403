# Báo Cáo Nhóm — Lab Day 08: RAG Pipeline

**Nhóm:** 04
**Thành viên:** 2A202600100_Lê Kim Dũng, 2A202600385_Ngô Gia Bảo, 2A202600395_Nguyễn Dương Ninh

## 1. Tóm tắt hệ thống RAG

Hệ thống RAG của nhóm được xây dựng trên cấu trúc pipeline tiêu chuẩn bao gồm Indexing, Retrieval và Generation, xử lý 5 tài liệu nghiệp vụ (SOP, Policy, FAQ) cho các bộ phận IT, HR, CS.

- **Indexing:** Sử dụng token-based chunking (500 tokens, overlap 50) và mô hình text-embedding-3-small (hoặc local) để lưu vào Vector Database ChromaDB. Chunks được làm giàu bằng metadata (source, section, effective_date) để phục vụ quá trình filtering.
- **Retrieval:** Tích hợp Hybrid retrieval kết hợp Reciprocal Rank Fusion (RRF) để kết hợp kết quả từ Semantic Search (Dense) và Keyword Search (Sparse - BM25). Đồng thời, nhóm ứng dụng Cross-encoder reranking để chắt lọc Top-3 chunks mang nhiều ý nghĩa và cung cấp context chính xác nhất.
- **Generation:** Sử dụng model gpt-4o-mini với prompt được tối ưu Grounding Strictness (yêu cầu mô hình trả lời "Tôi không biết" nếu context không chứa thông tin) và luôn trích dẫn nguồn văn bản (Citations).

## 2. Các quyết định kỹ thuật quan trọng của nhóm

Nhóm đã đưa ra những quyết định kỹ thuật quan trọng để nâng cao hiệu suất của hệ thống:

*   **Lựa chọn Hybrid Retrieval thay cho Dense Retrieval:** Nhóm nhận ra rằng cấu hình Dense Baseline làm mất các từ khóa kỹ thuật quan trọng như `ERR-403`, tên hệ thống đặc thù và các alias. Do đó, việc triển khai Hybrid giúp bù đắp khiếm khuyết của Dense, nâng điểm Context Recall từ 4.5 lên 5.0 đối với các query kỹ thuật mà không làm mất tính bao quát ngữ nghĩa.
*   **Áp dụng Cross-Encoder Reranker:** Mặc dù không đem lại cải thiện lớn so với riêng Hybrid trong tập dữ liệu nhỏ (kết quả baseline/variant context recall tương đồng do dense search vốn đã bắt được tài liệu), tuy nhiên Reranking là một quyết định chốt yếu để đảm bảo độ tin cậy khi tài liệu doanh nghiệp mở rộng quy mô (Scale-up) với nhiều đoạn chunk mang ý nghĩa chồng chéo.
*   **LLM Configuration:** Cố định `Temperature = 0` thay vì giá trị phổ thông (0.7) giúp mô hình tuân thủ tuyệt đối Instruction Prompt, hỗ trợ đánh giá tự động nhất quán khi triển khai LLM-as-Judge trong `eval.py`. Tối ưu prompt bắt buộc "Hãy trả lời 'tôi không biết' (abstain) dựa trên tài liệu" là bước ngoặc giúp tránh hallucination trong câu gq07.

## 3. Khó khăn lớn nhất và cách giải quyết

Khó khăn lớn nhất là việc xử lý các từ khóa viết tắt, biệt danh kỹ thuật (`SLA P1`, `ERR-403-AUTH`) và sự chồng chéo về quyền hạn trong tài liệu `access-control-sop.md`. Cả Retriever và LLM đôi khi nhầm lẫn trong việc cấp quyền Level 3 vs Level 2.

**Giải pháp của nhóm:**
Nhóm đã implement thành công Hybrid search kết hợp BM25 (thuần tìm từ khóa) để đảm bảo các chunk đề cập đến `ERR-403-AUTH` được nổi lên top đầu. Ngoài ra, việc tinh chỉnh prompt yêu cầu mô hình đọc kỹ và không đưa ra suy luận cá nhân đã cải thiện Faithful / Relevance.

## 4. Kế hoạch trong tương lai

Nếu có thêm thời gian, nhóm sẽ ứng dụng Query Transformation (chẳng hạn như HyDE hoặc Query decomposition) đối với các câu phức (ví dụ câu hỏi bao hàm cả hoàn tiền Flash sale và đổi trả hàng VIP) để model tự chiết xuất các ý nhỏ trước khi Retrieve.
