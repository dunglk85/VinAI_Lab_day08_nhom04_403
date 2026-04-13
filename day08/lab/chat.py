import os
import sys
from rag_answer import rag_answer

def run_chat_interface():
    print("="*60)
    print("🤖 RAG Chatbot - Lab Day 08")
    print("Gõ 'exit' hoặc 'quit' để thoát.")
    print("="*60)
    
    # Bạn có thể chọn chế độ mặc định ở đây (dense hoặc hybrid)
    # Khuyên dùng hybrid vì có hỗ trợ rerank và bắt keyword tốt hơn
    retrieval_mode = "hybrid" 
    use_rerank = True

    while True:
        try:
            query = input("\nBạn: ")
            if query.lower().strip() in ['exit', 'quit']:
                print("Tạm biệt!")
                break
                
            if not query.strip():
                continue
                
            print("Đang tìm kiếm và tạo câu trả lời...")
            # Gọi hàm RAG pipeline chúng ta đã xây dựng
            result = rag_answer(
                query, 
                retrieval_mode=retrieval_mode,
                use_rerank=use_rerank,
                verbose=False # Set thành True nếu bạn muốn xem các chunk được retrieve
            )
            
            print(f"\n🤖 Bot: {result['answer']}")
            if result['sources']:
                print(f"📚 Nguồn tham khảo: {', '.join(result['sources'])}")
                
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    run_chat_interface()
