# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 13/04/2026  
**Config:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = gpt-4o-mini
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.50 /5 |
| Answer Relevance | 4.60 /5 |
| Context Recall | 5.00 /5 |
| Completeness | 4.00 /5 |

**Câu hỏi yếu nhất (điểm thấp):**
> q07 (Approval Matrix) - Điểm Faithfulness thấp (3/5) vì model cố gắng giải thích dù context chỉ có một dòng ghi chú ngắn.
> q04 (Digital Refund) - Điểm Completeness thấp (3/5) vì liệt kê thiếu các ngoại lệ khác.

- [x] Indexing: Preprocessor bỏ lỡ dòng "Ghi chú" ở đầu file (Đã fix)
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [x] Generation: Model có xu hướng trả lời quá ngắn gọn (SLA P1)

---

## Variant 1 (Sprint 3)

**Ngày:** 13/04/2026  
**Biến thay đổi:** Hybrid Retrieval + Rerank  
**Lý do chọn biến này:**
> Thử nghiệm kết hợp Sparse (keyword) để xử lý mã lỗi và Alias tốt hơn. Rerank giúp chọn lọc Top 3 từ Top 10 tốt hơn.

**Config thay đổi:**
```
retrieval_mode = "hybrid"   
use_rerank = True
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.50/5 | 4.50/5 | 0.00 |
| Answer Relevance | 4.60/5 | 4.10/5 | -0.50 |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 4.00/5 | 3.40/5 | -0.60 |

**Nhận xét:**
> - Variant 1 (Hybrid + Rerank) không cho thấy sự cải thiện rõ rệt so với Baseline, thậm chí điểm Completeness giảm mạnh ở một số câu (q01, q08).
> - Lý do: Rerank đôi khi loại bỏ các chunk có chứa thông tin bổ sung quan trọng, dẫn đến câu trả lời bị cụt (Completeness giảm).
> - Với q09 và q10 (vấn đề abstain), cả hai đều thực hiện tốt nhưng điểm Relevance/Completeness bị thấp do cách LLM-Judge chấm điểm khi model trả lời "Tôi không biết".

**Kết luận:**
> - Không tối ưu hơn Baseline. Trong bài toán này, Dense Retrieval đơn thuần kết hợp với Preprocessing tốt (đã fix lỗi bỏ sót alias) là đủ hiệu quả.
> - Rerank cần tuning thêm về prompt hoặc sử dụng Cross-Encoder chuyên dụng thay vì LLM Reranking để tránh mất thông tin.

---

## Variant 2 (nếu có thời gian)

**Biến thay đổi:** Query Transform / Query Expansion
**Config:**
```
retrieval_mode = "query_transform"
top_k_search = 10
top_k_select = 3
use_rerank = True
```

**Scorecard Variant 2:**
| Metric | Baseline (Dense) | Variant 1 (Hybrid) | Variant 2 (Query Transform) | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | 4.50/5 | 4.50/5 | 4.50/5 | Tie |
| Answer Relevance | 4.60/5 | 4.10/5 | 4.20/5 | Baseline |
| Context Recall | 5.00/5 | 5.00/5 | 5.00/5 | Tie |
| Completeness | 4.00/5 | 3.40/5 | 3.20/5 | Baseline |

**Phân tích Hybrid vs Query Transform:**
> Cả 2 phương pháp nâng cao (Hybrid và Query Transform) đều có *Context Recall tuyệt đối (5.0)*, chứng tỏ việc lấy thông tin không còn là vấn đề sau khi fix lỗi Preprocessing.
> Tuy nhiên, **Query Transform** giúp điểm *Relevance* nhỉnh hơn một chút so với *Hybrid* (4.20 vs 4.10) nhờ vào việc Prompt viết lại câu hỏi có chèn đủ từ khóa ngữ cảnh. 
> Bù lại, điểm *Completeness* của Variant 2 thấp nhất (3.20) vì câu hỏi bị biến đổi đôi khi làm model thay đổi luôn trọng tâm câu trả lời (hoặc trả lời quá ngắn gọn). Nhìn chung, trong dataset này, Baseline Dense đơn giản vẫn đang là lựa chọn ổn định nhất.

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Lỗi Preprocessing (bỏ lỡ metadata/chú thích ở header) và lỗi mismatch dimension khi thay đổi embedding model mà không reset vector store.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > **Quality of Indexing (Preprocessing + Metadata)**: Việc đảm bảo model "thấy" được thông tin alias (Approval Matrix) quan trọng hơn việc đổi thuật toán retrieval.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Thử nghiệm điều chỉnh **Top-k select** lên cao hơn (ví dụ 5 thay vì 3) để tránh hiện tượng "Lost in the Middle" nhưng vẫn giữ đủ độ phủ thông tin cho Completeness.
