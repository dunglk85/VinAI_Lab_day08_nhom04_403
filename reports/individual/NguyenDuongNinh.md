# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Duong Ninh
**Vai trò trong nhóm:** Tech Lead / Document Owner
**Ngày nộp:** 13/04/2026
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)
Trong RAG Pipeline Lab (Day 08), tôi đóng vai trò là Tech Lead và Retrieval Owner, chủ yếu tập trung vào các hạng mục từ Sprint 2 và Sprint 3. Cụ thể, sau khi pipeline cơ bản hoàn thiện, tôi phân tích điểm thi và nhận ra hệ thống gặp vấn đề trong việc tra cứu các câu hỏi có chứa từ khóa kỹ thuật (ví dụ `ERR-403`), mà một mô hình Dense đơn thuần dễ dàng bỏ qua. Do đó, tôi đã triển khai kỹ thuật Hybrid Retrieval kết hợp Sparse (BM25) để thu thập chính xác các keyword và dùng Vector Search (Dense) để lọc thông tin có ý nghĩa tương đồng. Đồng thời, tôi tích hợp Cross-Encoder Reranker để thiết lập lại điểm Relevance chính xác cho Top-3 documents tốt nhất, giúp Generation của tác nhân đồng đội làm trong Sprint 2 dễ dàng hơn. Công việc của tôi kết nối chặt chẽ với Indexing từ Sprint 1 (với cách Metadata phân rã) và cung cấp đầu vào chuẩn xác nhất cho LLM Generator.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)
Điều khiến tôi thực sự hiểu rõ hơn sau Lab này là giới hạn của Semantic/Dense Search và sự bù đắp tuyệt vời của Hybrid Retrieval qua phương pháp Reciprocal Rank Fusion (RRF). Trước đây, tôi lầm tưởng rằng Vector Base luôn xuất sắc trong mọi trường hợp do tính "hiểu tương đồng ngữ nghĩa", tuy nhiên mô hình Embedding không phải lúc nào cũng ánh xạ tốt các keyword khô khan, mã lỗi (ERR-403, P1-SLA) - những thứ phải tra cứu chính xác. Nhờ triển khai Sparse/BM25 kết hợp với sự gán trọng số đồng nhất thông qua RRF, giờ đây tôi đã nắm rõ cách dung hòa kết quả tìm kiếm theo ngữ nghĩa và theo từ khóa. Bên cạnh đó, việc thực thi LLM-as-a-Judge cung cấp một vòng phản hồi Evaluation Loop tuyệt vời, giúp tôi liên tục tracking và so sánh các phiên bản theo A/B testing mà không phụ thuộc vào cảm tính chấm điểm thủ công.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)
Tôi hoàn toàn ngạc nhiên về tầm ảnh hưởng của Cross-encoder Reranker trong quy mô Dataset nhỏ. Giả thuyết ban đầu của tôi là việc tích hợp Cross-encoder (Sử dụng model ms-marco) sẽ loại bỏ triệt để mọi lỗi hallucination do tìm nhầm tài liệu. Nhưng trên thực tế, sau khi chạy Pipeline tự động chấm điểm trên tập Test gồm 10 Document, điểm số Context Recall giữa Hybrid thông thường và Hybrid+Reranker hầu như không giãn cách thêm so với mong đợi (vẫn giữ vững mức tỷ lệ Context Recall 5.0 rất cao). Mặt mặt khác, khó khăn lớn nhất mà tôi cần xử lý là thiết kế Prompt ép (Grounding Strictness). LLM GPT-4-mini thường xuyên có xu hướng "bốc" thông tin từ pre-trained model knowledge khi gặp câu hỏi không có trong Document (gq07). Tôi mất rất nhiều thời gian test prompt engineer để ép mức độ tuân thủ Abstain (Tôi không biết) cho đến khi thành công cải thiện điểm số ở các câu out-of-context.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** [q09] Lại thêm trường hợp nhân viên gặp lỗi ERR-403 khi login, lỗi này là gì và làm sao để khắc phục?

**Phân tích:**
Trong phiên bản Baseline (với Dense Search), nhóm trả về câu trả lời với mức điểm Faithfulness và Relevance cực kỳ thấp (1/5) trong khi Completeness đo được cũng chỉ đạt 1. Hệ thống đã cung cấp đáp án mơ hồ ("Tôi không biết" hoặc trả lời từ kiến thức nạp sẵn) thay vì cung cấp hướng dẫn của công ty. Lỗi ở đây hoàn toàn xuất phát từ module **Retrieval**. Thuật toán Dense Retrieval không bắt được cụm từ đặc biệt mang tính technical `"ERR-403"`, dẫn tới việc context truyền vào không mang theo bất kỳ đoạn tài liệu xử lý sự cố đăng nhập nào. Khi được nâng cấp lên thẻ Variant (Có sử dụng Hybrid - kết hợp BM25 Keyword Search), Pipeline ngay lập tức cải thiện điểm số. Faithfulness và Relevant đã được cải thiện lên đến 5/5 vì Sparse match dễ dàng tra ra đoạn chứa chính xác Text `ERR-403-AUTH`, giúp cung cấp context vô cùng chính xác để tiến hành Generation. Việc hiểu rõ failure point này giúp tôi củng cố quyết định thêm Hybrid Retrieval là hoàn toàn đúng đắn.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)
Nếu có thêm thời gian, tôi sẽ thử ứng dụng kỹ thuật **Query Transformation (HyDE - Hypothetical Document Embeddings)**. Lý do là qua quá trình Eval, tôi nhận thấy các câu hỏi của người dùng thường ngắn và thiếu context ngữ cảnh nhưng lại đòi hỏi giải pháp chuyên sâu (như câu hỏi q04, q10 về chính sách hoàn tiền). HyDE sẽ giúp LLM vẽ ra một phiên bản tài liệu giả định chi tiết trước khi nhúng thành Vector, kỳ vọng sẽ cải thiện độ trúng đích của Dense Retrieval đối với các User queries quá gọn lỏn.