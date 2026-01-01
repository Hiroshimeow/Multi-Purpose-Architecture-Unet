# Cấu trúc thư mục dự án

Đây là cấu trúc thư mục đã được tổ chức lại của dự án TABS-UNet.

```
.
├── GEMINI.md
├── README.md
├── requirements.txt
├── configs/             # Chứa các file cấu hình (.yaml) cho mỗi lần chạy thí nghiệm.
├── data/
├── datasets/            # Chứa các bộ dữ liệu siêu phổ đã được xử lý cho các phương pháp so sánh.
├── docs/                # Chứa các tài liệu, báo cáo, và các file markdown liên quan đến dự án.
├── logs/                # Chứa các file log chi tiết từ các lần chạy thí nghiệm.
├── results/             # Chứa các file kết quả tổng hợp, ví dụ như file CSV so sánh.
├── scripts/             # Chứa tất cả các script Python để chạy và quản lý dự án.
│   ├── analysis/        # Scripts để phân tích (tính FLOPs, stats model).
│   ├── generation/      # Scripts để tạo dữ liệu hoặc câu lệnh.
│   └── verification/    # Scripts để kiểm tra (ví dụ: tham số).
├── src/                 # Chứa mã nguồn chính của mô hình TABS-UNet và các thành phần liên quan.
└── thesis/              # Chứa các file liên quan đến luận văn tốt nghiệp (file LaTeX, hình ảnh).
```

---

# Kế hoạch phát triển Luận văn Tốt nghiệp: TABS-UNet

Dựa trên các phản hồi từ hội đồng review (IEVC 2026) và mục tiêu nâng cấp bài báo thành một luận văn tốt nghiệp hoàn chỉnh, kế hoạch phát triển được đề xuất như sau. Kế hoạch này được cấu trúc thành các giai đoạn chính để giải quyết từng nhóm vấn đề một cách có hệ thống.

---

### Giai đoạn 1: Củng cố Nền tảng Lý thuyết và Vị thế của Đề tài (Addressing Reviewers 3 & 4)

Mục tiêu của giai đoạn này là khắc phục những điểm yếu lớn nhất về phần tổng quan tài liệu (related work) và làm nổi bật tính mới của phương pháp.

*   **Task 1.1: Mở rộng tổng quan về các phương pháp Lightweight cho HSI:**
    *   **Hành động:** Nghiên cứu và bổ sung các mô hình lightweight dựa trên Transformer gần đây cho bài toán phân vùng ảnh siêu phổ (HSI segmentation).
    *   **Lý do:** Giải quyết trực tiếp góp ý của Reviewer 3 về việc so sánh với các mô hình Transformer.

*   **Task 1.2: Cập nhật và đa dạng hóa tài liệu tham khảo:**
    *   **Hành động:** Tìm kiếm và bổ sung các bài báo khoa học từ 2-3 năm trở lại đây về các lĩnh vực: band selection, lightweight image segmentation, và ứng dụng HSI trong xe tự lái.
    *   **Lý do:** Giải quyết chỉ trích của Reviewer 4 về việc các tài liệu tham khảo đã cũ.

*   **Task 1.3: Tích hợp các bài báo được gợi ý:**
    *   **Hành động:** Đọc và đưa bài báo "Concrete Autoencoders for Differentiable Feature Selection and Reconstruction" (Abid et al., ICML 2019) vào phần related work.
    *   **Lý do:** Giải quyết gợi ý trực tiếp từ Reviewer 1, cho thấy sự nghiêm túc trong việc tiếp thu góp ý.

*   **Task 1.4: Viết lại phần Introduction và Related Work:**
    *   **Hành động:** Sau khi thu thập đủ tài liệu, viết lại hoàn toàn phần Introduction để làm rõ "research gap" (khoảng trống nghiên cứu) và định vị TABS-UNet một cách thuyết phục hơn. Tái cấu trúc phần Related Work để có một cái nhìn tổng quan, có chiều sâu và làm nổi bật sự khác biệt của TABS-UNet so với các phương pháp đã có.
    *   **Lý do:** Giải quyết cốt lõi vấn đề của Reviewer 4 về "novelty" và "discussion on related work".

---

### Giai đoạn 2: Làm rõ và Bảo vệ Luận điểm Kỹ thuật (Addressing Reviewer 1)

Giai đoạn này tập trung vào việc trả lời câu hỏi kỹ thuật quan trọng từ Reviewer 1.

*   **Task 2.1: Giải thích rõ lý do sử dụng Straight-Through Estimator (STE):**
    *   **Hành động:** Trong chương "Methodology", thêm một đoạn giải thích chi tiết tại sao cần kết hợp Gumbel-Softmax với STE. Luận điểm có thể tập trung vào việc STE giúp ổn định quá trình huấn luyện trong kiến trúc end-to-end của TABS-UNet, hoặc giải quyết một vấn đề cụ thể nào đó (ví dụ: gradient variance) mà chỉ dùng Gumbel-Softmax không xử lý được.
    *   **Lý do:** Trả lời trực tiếp và làm thỏa mãn thắc mắc của Reviewer 1.

---

### Giai đoạn 3: Mở rộng và Đào sâu Thực nghiệm (Addressing Reviewers 1 & 3)

Mục tiêu là tăng cường độ tin cậy và tính thuyết phục của kết quả thực nghiệm thông qua các thí nghiệm bổ sung.

*   **Task 3.1: Thí nghiệm về sự ổn định của việc chọn băng tần (Band Stability Analysis):**
    *   **Hành động:** Thiết kế và chạy lại thí nghiệm nhiều lần với các `random seed` khác nhau (ví dụ: 5-10 seeds). Ghi lại các bộ băng tần được chọn ở mỗi lần chạy. Lập bảng hoặc biểu đồ để phân tích và báo cáo mức độ ổn định (ví dụ: bao nhiêu % các băng tần được chọn lặp lại qua các lần chạy).
    *   **Lý do:** Giải quyết trực tiếp yêu cầu của Reviewer 3 về "analysis of band stability", một luận điểm rất giá trị để chứng minh sự mạnh mẽ (robustness) của TABS.

*   **Task 3.2: Thí nghiệm với nhiều số lượng băng tần được chọn (Ablation Study on `k`):**
    *   **Hành động:** Chạy lại các thí nghiệm với các giá trị `k` khác nhau ngoài 5, ví dụ `k = 3, 7, 9`. Báo cáo kết quả về mIoU và FPS (hoặc FLOPs) cho từng giá trị `k`.
    *   **Lý do:** Trả lời câu hỏi của Reviewer 1 và cung cấp một cái nhìn toàn diện hơn về sự đánh đổi giữa hiệu năng và độ chính xác.

*   **Task 3.3: So sánh với các phương pháp State-of-the-Art (SOTA) mới:**
    *   **Hành động:** Dựa trên kết quả từ Giai đoạn 1, chọn ra 1-2 phương pháp SOTA gần đây và phù hợp nhất để so sánh trực tiếp với TABS-UNet trên cùng một bộ dữ liệu.
    *   **Lý do:** Giải quyết góp ý của Reviewer 4 về việc thiếu so sánh với các phương pháp SOTA.

---

### Giai đoạn 4: Hoàn thiện Luận văn

*   **Task 4.1: Cập nhật kết quả và thảo luận:**
    *   **Hành động:** Tạo các bảng biểu, hình ảnh mới cho các kết quả từ Giai đoạn 3. Viết lại chương "Results and Discussion" để phân tích sâu các kết quả mới, bao gồm sự ổn định của băng tần, sự ảnh hưởng của `k`, và vị thế của TABS-UNet so với SOTA.
*   **Task 4.2: Hoàn thiện bản thảo:**
    *   **Hành động:** Cập nhật lại toàn bộ tài liệu tham khảo, rà soát lại toàn bộ luận văn để đảm bảo tính nhất quán, không có lỗi chính tả, và văn phong chuyên nghiệp, học thuật.

---
---

## PHỤ LỤC A: CHI TIẾT HÓA KẾ HOẠCH VÀ TÍCH HỢP THÔNG TIN

Phần này sẽ chi tiết hóa các hành động cần thực hiện dựa trên cấu trúc file trong `@thesis/` và kết quả tìm kiếm trên internet.

### 1. Phân tích Cấu trúc và Nội dung cần sửa đổi

Dù không đọc được trực tiếp nội dung, dựa trên cấu trúc file, đây là những nơi cần tập trung sửa đổi:

*   **`title.tex`**: Cần xem xét lại tiêu đề để phản ánh đúng tầm vóc của một luận văn, có thể mở rộng hơn so với tên bài báo.
*   **`h20074_abst.tex` (Abstract)**: Cần được viết lại sau khi hoàn thành các thí nghiệm bổ sung. Abstract mới phải tóm tắt được cả các kết quả về **tính ổn định (stability)** và **ảnh hưởng của k (ablation study on k)**, đây là những đóng góp mới quan trọng.
*   **`chapters/introduction.tex` (Giả định)**: Đây là phần cần **viết lại nhiều nhất**. Phải làm nổi bật được khoảng trống nghiên cứu (research gap), định vị TABS-UNet một cách rõ ràng so với các hướng tiếp cận CNN, Transformer, và các phương pháp band selection khác.
*   **`chapters/related_work.tex` (Giả định)**: Cần **viết lại và mở rộng đáng kể**.
    *   Tạo một subsection mới về **"Lightweight Transformer-based Models for HSI"**.
    *   Tích hợp bài báo "Concrete Autoencoders" và thảo luận về mối liên hệ với TABS-UNet.
    *   Cập nhật các phương pháp SOTA mới nhất (2023-2025).
*   **`chapters/methodology.tex` (Giả định)**: Thêm mục **2.1** để giải thích chi tiết về sự kết hợp Gumbel-Softmax và STE.
*   **`chapters/experiments.tex` (Giả định)**: Thêm các subsection mới để trình bày kết quả từ **Giai đoạn 3**:
    *   "Band Selection Stability Analysis" (Task 3.1)
    *   "Ablation Study on the Number of Selected Bands" (Task 3.2)
    *   "Comparison with State-of-the-Art Methods" (Task 3.3)
*   **`references.bib`**: File này sẽ phình to đáng kể. Cần bổ sung tất cả các paper mới đã được nghiên cứu ở Giai đoạn 1.

### 2. Tích hợp Kết quả Tìm kiếm

#### a. Lightweight Transformer Models for HSI (Giải quyết Review 3)

Kết quả tìm kiếm cho thấy một số hướng tiếp cận và model nổi bật mà bạn nên đưa vào phần Related Work để thảo luận và so sánh:

*   **Các Model tiêu biểu:**
    *   **ResFormer:** Sử dụng Transformer với độ phức tạp tuyến tính (linear complexity), phù hợp cho ảnh có độ phân giải cao.
    *   **Efficient Transformer (dựa trên Swin):** Giảm tải tính toán cho Swin Transformer, tập trung vào tăng tốc độ inference.
    *   **GATrans (Generative Adversarial Transformer):** Hướng tới hiệu quả cao với kích thước model cực nhỏ.
    *   **HSTR (Hyperspectral 3D Transformer):** Xử lý cả thông tin không gian và quang phổ (spatial-spectral) một cách đồng thời.
    *   **MSA-LWFormer:** Kết hợp CNN, Attention và Transformer, sử dụng Fast Fourier Transform (FFT) để giảm độ phức tạp.
*   **Chiến lược chung cần thảo luận:**
    *   **Hybrid Architectures (CNN + Transformer):** Đây là hướng phổ biến, kết hợp khả năng trích xuất đặc trưng local của CNN và khả năng nắm bắt ngữ cảnh global của Transformer. TABS-UNet của bạn có thể được định vị trong bối cảnh này.
    *   **Linear Complexity Attention:** Nhiều model đang tìm cách giảm độ phức tạp bậc hai O(n^2) của self-attention. Đây là một điểm mạnh cần nhấn mạnh nếu TABS-UNet có ưu thế về mặt này.

**Hành động cụ thể:** Đưa các model và chiến lược này vào `chapters/related_work.tex` để cho thấy bạn đã có một cái nhìn toàn diện về lĩnh vực.

#### b. "Concrete Autoencoders for Differentiable Feature Selection" (Giải quyết Review 1)

**Ghi chú:** Tiếc là các công cụ tìm kiếm của tôi liên tục thất bại trong việc truy xuất bài báo này. Đây có thể là một hạn chế tạm thời của tool.

**Đề xuất:**
1.  **Vui lòng bạn cung cấp link PDF trực tiếp hoặc bản tóm tắt của bài báo này nếu có.**
2.  Tuy nhiên, dựa trên tiêu đề, có thể suy luận: Bài báo này đề xuất một kiến trúc autoencoder sử dụng "Concrete Distribution" (một tên gọi khác của Gumbel-Softmax) để thực hiện feature selection một cách khả vi (differentiable).
3.  **Điểm cần thảo luận trong luận văn:** So sánh phương pháp của họ và TABS-UNet.
    *   Họ dùng autoencoder để chọn feature, còn bạn dùng một selection layer trong một kiến trúc UNet cho bài toán segmentation. Mục tiêu cuối cùng khác nhau (reconstruction vs. segmentation).
    *   Họ có cần STE không? Nếu không, tại sao kiến trúc của họ cho phép điều đó còn của bạn thì không (hoặc ngược lại)? Phân tích sự khác biệt này sẽ giúp làm rõ luận điểm kỹ thuật của bạn (Task 2.1).

Việc phân tích và so sánh này sẽ là một sự bổ sung rất giá trị cho chương `Methodology` hoặc `Related Work`.