# Thuyết Minh Kỹ Thuật 10 Câu Hỏi Phòng Vệ — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Phạm Danh Tuấn Dũng  
**MSSV:** 2A202601978  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  

---

### Câu 1: Tại sao phải dùng Conservative Coreference Resolution thay vì để LLM tự do phân giải?
**Trả lời:**
- *Bản chất kỹ thuật:* Các mô hình LLM khi được giao nhiệm vụ phân giải đại từ tự do thường có xu hướng gán đại từ cho thực thể gần nhất trong câu trước đó (recency bias) hoặc thực thể nổi tiếng nhất (salience bias).
- *Nguy cơ:* Nếu trong một đoạn tin tức có 2 công ty được nhắc đến (ví dụ: *"Amazon announced partnerships with AI startup Cohere. The company also introduced AWS HealthScribe"*), LLM phân giải tự do có thể gán nhầm *"The company"* thành *"Cohere"*. Bước trích xuất quan hệ tiếp theo sẽ tạo ra một cạnh sai `(Cohere)-[:DEVELOPED]->(AWS HealthScribe)`.
- *Giải pháp:* Thiết kế prompt có ràng buộc bảo thủ (Conservative Prompting) chỉ cho phép phân giải khi antecedent được xác thực rõ ràng, bảo toàn nguyên vẹn số liệu, tên riêng và xuất danh sách `unresolved_mentions` cho các trường hợp mơ hồ thay vì suy đoán liều lĩnh.

---

### Câu 2: Ý nghĩa của Schema Guard & Allowlist trong việc kiểm soát chất lượng Graph?
**Trả lời:**
- *Bản chất kỹ thuật:* Trong Open Information Extraction (OpenIE), LLM có thể sinh ra vô số loại nhãn thực thể và quan hệ biến thể (như `developed`, `built`, `created`, `launched`, `engineered`). Điều này làm bùng nổ không gian quan hệ (relation explosion) và làm suy giảm khả năng thực thi truy vấn đồ thị Cypher.
- *Giải pháp:* Hệ thống giới hạn nghiêm ngặt 3 loại Node (`Company`, `Person`, `Technology`) và 8 loại Relation chuẩn hóa (`ACQUIRED`, `DEVELOPED`, `INVESTED_IN`, `FOUNDED`, `WORKED_AT`, `PARTNERED_WITH`, `USES`, `LEADS`). Bất kỳ quan hệ nào không nằm trong allowlist đều bị lọc bỏ ngay tại tầng tiền xử lý của Module 2.

---

### Câu 3: Tại sao cần Edge Provenance Tracking và hậu quả của việc thiếu Provenance là gì?
**Trả lời:**
- *Bản chất kỹ thuật:* Mỗi cạnh trong Knowledge Graph bắt buộc phải lưu trữ thuộc tính `source_chunk_id`, `published_date`, `evidence` (câu văn bằng chứng trực tiếp) và `confidence`.
- *Lợi ích:* 
  1. Cho phép GraphRAG tạo ra các câu trả lời có trích dẫn nguồn `[chunk_id=...]` minh bạch, giúp người dùng có thể đối chiếu và kiểm tra tính xác thực.
  2. Cho phép thực hiện các bộ lọc theo thời gian (Temporal filtering) khi xử lý Super-nodes.
- *Hậu quả khi thiếu:* Đồ thị trở thành "hộp đen" không thể kiểm toán (unauditable graph). Nếu một cạnh sai lọt vào đồ thị, không có cách nào truy vết chunk gốc để gỡ bỏ hoặc sửa lỗi.

---

### Câu 4: Phân tích cơ chế Entity Resolution: Tại sao cần kết hợp Manual Aliases + Vector ANN + Lexical Guard?
**Trả lời:**
- *Bản chất:* 
  - **Manual Aliases:** Xử lý các từ viết tắt chuẩn công nghiệp mà vector embedding khó phân biệt (ví dụ: `MSFT -> Microsoft`, `GOOG -> Google`, `HPE -> Hewlett Packard Enterprise`).
  - **Vector ANN (Faiss IndexFlatIP $\ge 0.90$):** Bắt các biến thể ngữ nghĩa gần nhau như `Microsoft Corp`, `Microsoft Corporation`.
  - **Lexical Guard (SequenceMatcher $\ge 0.72$):** Là chốt chặn an toàn ngăn ngừa các trường hợp embedding gần nhau nhưng là 2 thực thể khác biệt (như `Sam Altman` vs `Steve Altman`, `Apple` vs `Apple Music`).
- *Cấu trúc Disjoint-Set Union-Find (DSU):* Đảm bảo tính bắc cầu khi gom cụm các thực thể ($A \equiv B$ và $B \equiv C \Rightarrow A \equiv C$) với độ phức tạp gần như tuyến tính $O(\alpha(N))$.

---

### Câu 5: Super-node Mitigation: Tại sao chọn cắt tỉa theo `published_date DESC LIMIT 50`?
**Trả lời:**
- *Bản chất Super-node:* Trong đồ thị tri thức, phân bố bậc của các nút tuân theo luật lũy thừa (Power-law distribution). Các thực thể như `OpenAI` hay `Microsoft` có bậc kết nối cực lớn (Degree $> 100$).
- *Nguy cơ:* Nếu không giới hạn, khi duyệt BFS 2-hop qua một Super-node, số lượng cạnh thu thập sẽ bùng nổ lên hàng ngàn, vượt quá context window của LLM và gây tràn bộ nhớ.
- *Lý do chọn cắt tỉa theo ngày xuất bản:* Trong miền tin tức công nghệ, các sự kiện mới nhất có trọng số thông tin cao hơn các sự kiện cũ. Lấy 50 cạnh mới nhất giúp giữ ngữ cảnh trả lời luôn tươi mới và giữ kích thước prompt trong tầm kiểm soát an toàn.

---

### Câu 6: So sánh cơ chế tìm kiếm của Flat RAG vs Hybrid GraphRAG?
**Trả lời:**
- *Flat RAG:* Sử dụng Dense Vector Embedding và tìm kiếm $k$ đoạn văn bản có khoảng cách cosine nhỏ nhất với câu hỏi. 
  - *Ưu điểm:* Tốc độ cực nhanh, đơn giản, chi phí tính toán thấp.
  - *Nhược điểm:* Mù thông tin cấu trúc; không thể liên kết các đoạn văn bản rời rạc khi không có từ khóa trùng lặp ngữ nghĩa trực tiếp (Semantic Disconnection).
- *Hybrid GraphRAG:* Trích xuất seed entities từ câu hỏi, tìm kiếm nút trong đồ thị, duyệt BFS $N$-hop thu thập mạng lưới quan hệ có kiểm soát, sau đó kết hợp cả cấu trúc quan hệ (`=== GRAPH ===`) và tài liệu vector (`=== VECTOR ===`) để làm giàu prompt cho LLM.
  - *Ưu điểm:* Giải quyết xuất sắc các bài toán đa bước (Multi-hop Reasoning) và tổng hợp toàn cảnh.

---

### Câu 7: Tại sao GraphRAG có Token Usage và Latency cao hơn Flat RAG?
**Trả lời:**
- *Về Latency:* GraphRAG trải qua quy trình đa tầng: LLM/Fuzzy Seed Extraction $\rightarrow$ Graph Traversal $\rightarrow$ Edge Linearization $\rightarrow$ Vector Search $\rightarrow$ LLM Synthesis. Thời gian xử lý trung bình tăng thêm từ $0.3\text{s}$ đến $0.8\text{s}$.
- *Về Token Usage:* GraphRAG đính kèm toàn bộ danh sách các cạnh có cấu trúc (`Source [Type] -RELATION-> Target [Type] | date | chunk | evidence`) vào prompt bên cạnh các đoạn văn bản thô, làm tăng số lượng token đầu vào lên khoảng 40%–60%.

---

### Câu 8: Vai trò của LLM-as-a-Judge trong việc đánh giá hệ thống RAG?
**Trả lời:**
- *Hạn chế của metric truyền thống (BLEU, ROUGE):* Chỉ đo lường độ trùng lặp từ ngữ bề mặt (n-gram overlap), hoàn toàn bất lực trong việc đánh giá tính logic, tính bao quát và tính trung thực của các câu trả lời diễn đạt bằng câu chữ khác nhau.
- *Lợi thế của LLM-as-a-Judge:* Đánh giá ngữ nghĩa sâu theo 3 tiêu chí chuẩn công nghiệp:
  1. **Comprehensiveness:** Mức độ bao phủ đầy đủ tất cả các thực thể và khía cạnh yêu cầu.
  2. **Faithfulness:** Mức độ bám sát bằng chứng ngữ cảnh, không bịa đặt thông tin.
  3. **Multi-hop Reasoning:** Khả năng liên kết chính xác các mối quan hệ logic đa tầng.
  Kèm theo bản thuyết minh lý do chấm điểm (Judge Rationale) giúp dễ dàng kiểm toán và debug lỗi.

---

### Câu 9: Bonus Community Detection: Thuật toán Greedy Modularity mang lại giá trị gì?
**Trả lời:**
- *Bản chất kỹ thuật:* Sử dụng thuật toán `nx.algorithms.community.greedy_modularity_communities` để phân hoạch đồ thị tri thức thành các cụm (communities) có mật độ liên kết nội bộ cao.
- *Ứng dụng:* Cho phép thực hiện **Global Query Summarization** (tương tự kiến trúc GraphRAG của Microsoft Research): Tóm tắt trước nội dung của từng cụm cộng đồng (ví dụ: Cụm đối tác đám mây AI, Cụm bảo mật mạng Edge-to-cloud). Khi gặp câu hỏi vĩ mô cấp cao (ví dụ: *"Tổng kết các xu hướng M&A trong ngành bảo mật 2023"*), hệ thống chỉ cần đọc bản tóm tắt cộng đồng thay vì phải duyệt toàn bộ đồ thị.

---

### Câu 10: Bonus Self-Correction Retrieval: Cơ chế Fallback hoạt động ra sao?
**Trả lời:**
- *Cơ chế hoạt động:*
  1. Bước 1: Hệ thống truy xuất đồ thị ở phạm vi tiêu chuẩn 2-hop (`Hop 2`).
  2. Bước 2: Chạy bộ kiểm tra tính đầy đủ (Sufficiency Check) bằng một prompt LLM siêu nhẹ: *"Đoạn ngữ cảnh này có đủ thông tin để trả lời câu hỏi hay không?"*.
  3. Bước 3: Nếu `sufficient == True`, tiến hành sinh câu trả lời ngay. Nếu thiếu (`False`), tự động mở rộng lên `Hop 3`.
  4. Bước 4: Nếu `Hop 3` vẫn chưa đủ, kích hoạt cơ chế Fallback kết hợp `Hop 3 + Vector Search mở rộng (k=8)`.
- *Ý nghĩa:* Đảm bảo tỷ lệ trả lời thành công tối đa mà vẫn tiết kiệm chi phí token cho các câu hỏi đơn giản.
