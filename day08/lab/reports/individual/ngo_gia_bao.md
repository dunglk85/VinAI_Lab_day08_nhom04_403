# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Ngô Gia Bảo
**Vai trò trong nhóm:** Eval Owner
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong dự án lab này, tôi đảm nhận vai trò Eval Owner, chịu trách nhiệm chính trong việc đánh giá chất lượng của hệ thống RAG. Cụ thể, tôi đã thực hiện:
- **Xây dựng bộ Test Set:** Tôi đã thiết kế 10 câu hỏi kiểm thử bao phủ nhiều trường hợp khác nhau (factoid, synthesis, open-ended) dựa trên tập tài liệu chính sách của công ty.
- **Triển khai LLM-as-a-Judge:** Tôi đã thiết lập pipeline đánh giá tự động (scorecard) sử dụng LLM để chấm điểm các tiêu chí Faithfulness, Relevance, Context Recall và Completeness cho cả phiên bản Baseline và Hybrid Variant.
- **Phân tích kết quả và Lỗi:** Thông qua scorecard, tôi nhận diện các điểm yếu của hệ thống (như việc trả lời sai mã lỗi hay thiếu thông tin) và phối hợp với Retrieval Owner để chẩn đoán nguyên nhân, sau đó ghi chú lại trong bảng so sánh kết quả cũng như hỗ trợ điều chỉnh các pipeline.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Điều tôi hiểu rõ nhất sau lab này chính là **tầm quan trọng của Evaluation Loop (Vòng lặp đánh giá tự động)** và cách định lượng chất lượng RAG. Ban đầu, tôi đánh giá khá cảm tính, cho rằng chỉ cần đọc câu trả lời thấy "có vẻ đúng" là được. Tuy nhiên, khi sử dụng LLM-as-Judge, tôi nhận thấy các góc nhìn chi tiết hơn—đặc biệt là sự khác biệt giữa "trả lời có đúng và dựa vào context không (Faithfulness)" và "trả lời có đầy đủ không (Completeness)". Tôi cũng hiểu sâu sắc hơn về tính ưu việt của tự động hóa: thay vì mất hàng giờ đọc từng câu mỗi khi thay đổi prompt hay thuật toán, việc có sẵn một scorecard chạy tự động cho phép team tối ưu (iterate) cực kỳ nhanh chóng và tự tin chứng minh được sự cải thiện.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Khó khăn lớn nhất là **LLM-as-a-Judge đôi khi chấm điểm khá khắt khe và cần tinh chỉnh prompt (tiêu chí đánh giá)**. Trong những lần chạy đầu tiên, LLM Judge cho điểm Relevance thấp dù câu trả lời có liên quan, vì nó đòi hỏi câu trả lời phải bao trọn cả một số chi tiết phụ trong ngữ cảnh.

Điều làm tôi cực kỳ ngạc nhiên là khi so sánh giữa Baseline (chỉ Dense) và Variant (Hybrid + Reranking), **điểm của phiên bản Variant lại thấp hơn ở tiêu chí Completeness (3.20/5) và Relevance (4.10/5)** so với Baseline sau khi đã fix indexing (Completeness: 4.00, Relevance: 4.60). Việc Reranking vô tình loại bỏ một số chunk có ý nghĩa bổ sung (nhưng điểm số sematic relevance thấp), khiến Generation bị thiếu hụt ý tưởng. Điều này cho thấy với tập văn bản nhỏ, Dense Retrieval thuần có thể hoạt động ổn định và đầy đủ hơn.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** "Approval Matrix để cấp quyền hệ thống là tài liệu nào?" (q07)

**Phân tích từ góc độ Evaluation:**
Trước khi nhóm fix vấn đề tiền xử lý, hệ thống đạt điểm Faithfulness rất thấp (2/5) vì không tìm thấy tên "Approval Matrix", mô hình buộc phải bịa câu trả lời hoặc abstain nhầm do đoạn chú thích này từng bị coi là rác. 

Nhưng điều thú vị là trong đợt test Variant (Hybrid + Reranking), một lần nữa câu này lại bộc lộ hạn chế. Dù Hybrid Retrieval có thể bắt được keyword, nhưng bước LLM Reranking (nếu thiết lập prompt ranking hoặc cut-off chặt) lại vô tình đẩy chunk chứa chú thích alias đó xuống dưới cùng và rơi ra ngoài `top_k=3` được đưa vào context cuối, khiến câu trả lời bị hụt thông tin. Qua phân tích log và điểm số, tôi kiến nghị Retrieval Owner rằng **"không nên lạm dụng Rerank nếu nó gây rủi ro loại bỏ dữ liệu hiếm"**. Cuối cùng, bản Baseline sau fix parsing lại là bản trả lời câu này tốt và lấy điểm scorecard ổn định nhất.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ mở rộng Test Set từ 10 câu lên 30-50 câu để đảm bảo các điểm số metric mang tính đại diện thống kê cao hơn, chia làm các nhóm khó dễ rõ ràng (ví dụ: cần tổng hợp từ nhiều tài liệu/nhiều section). Ngoài ra, tôi cũng muốn nghiên cứu sử dụng các evaluation framework chuẩn công nghiệp như **Ragas** hay **TruLens** để đo lường tự động độ chính xác của Retrived Context chuyên sâu hơn thay vì tự code bằng tay kịch bản chấm điểm.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
