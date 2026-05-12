-- ==============================================================================
-- BƯỚC 0: KÍCH HOẠT EXTENSION AI (Bắt buộc cho Supabase / pgvector)
-- ==============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ==============================================================================
-- BƯỚC 1: KHỞI TẠO 2 SCHEMA (VÙNG CÁCH LY LOGIC)
-- ==============================================================================
CREATE SCHEMA IF NOT EXISTS library_core;
CREATE SCHEMA IF NOT EXISTS ai_engine;

-- ==============================================================================
-- BƯỚC 2: XÂY DỰNG PHÂN HỆ BACKEND (LIBRARY_CORE)
-- Nhiệm vụ: Xử lý nghiệp vụ Mượn/Trả, Quản lý sách, Lưu vết hành vi
-- ==============================================================================

-- 2.1 Bảng Người dùng (Giả lập, vì Supabase có auth.users, nhưng nên có bảng profile riêng)
CREATE TABLE IF NOT EXISTS library_core.users (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2.2 Bảng Ấn phẩm / Sách (Trái tim của thư viện)
CREATE TABLE IF NOT EXISTS library_core.publications (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    author VARCHAR(255),
    cover_image_url VARCHAR(500),
    available_copies INT DEFAULT 1,
    total_borrows INT DEFAULT 0,
    -- 2 Cột dành riêng cho AI cập nhật kết quả Pha 2 (Reduce):
    ai_summary TEXT,
    ai_target_audience VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2.3 Bảng Lịch sử Mượn sách
CREATE TABLE IF NOT EXISTS library_core.borrow_records (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES library_core.users(id) ON DELETE CASCADE,
    publication_id INT REFERENCES library_core.publications(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'BORROWED', -- BORROWED, RETURNED, OVERDUE
    borrow_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    return_date TIMESTAMP
);

-- 2.4 Bảng Lưu vết Hành vi (Thức ăn cho thuật toán Gợi ý ALS)
CREATE TABLE IF NOT EXISTS library_core.user_interactions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES library_core.users(id) ON DELETE CASCADE,
    publication_id INT REFERENCES library_core.publications(id) ON DELETE CASCADE,
    action_type VARCHAR(50) NOT NULL, -- 'VIEW', 'WISHLIST', 'BORROW'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tối ưu hóa truy vấn cho AI (Rất quan trọng khi quét dữ liệu tính điểm)
CREATE INDEX IF NOT EXISTS idx_user_interactions_scan
ON library_core.user_interactions(user_id, publication_id, action_type);


-- ==============================================================================
-- BƯỚC 3: XÂY DỰNG PHÂN HỆ AI (AI_ENGINE)
-- Nhiệm vụ: Phục vụ Semantic Search và Lưu trữ Tags linh hoạt
-- ==============================================================================

-- 3.1 Bảng Tags (Cơ chế Find-or-Create)
CREATE TABLE IF NOT EXISTS ai_engine.tags (
    id SERIAL PRIMARY KEY,
    -- Ràng buộc UNIQUE là BẮT BUỘC để lệnh ON CONFLICT DO NOTHING hoạt động
    name VARCHAR(100) UNIQUE NOT NULL
);

-- 3.2 Bảng Trung gian (Sách - Tags)
CREATE TABLE IF NOT EXISTS ai_engine.publication_tags (
    publication_id INT REFERENCES public.publications(id) ON DELETE CASCADE,
    tag_id INT REFERENCES ai_engine.tags(id) ON DELETE CASCADE,
    -- Khóa chính kép đảm bảo 1 sách không bị gán 2 lần cùng 1 tag
    PRIMARY KEY (publication_id, tag_id)
);

-- 3.3 Bảng Vector Ngữ nghĩa (Lõi tìm kiếm của pgvector)
CREATE TABLE IF NOT EXISTS ai_engine.publication_vectors (
    id SERIAL PRIMARY KEY,
    publication_id INT REFERENCES public.publications(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    -- Model SBERT của Bách Khoa xuất ra vector 768 chiều
    embedding vector(768)
);

-- BÙA HỘ MỆNH CHỐNG SẬP CPU: Đánh chỉ mục HNSW cho hệ thống tìm kiếm
CREATE INDEX IF NOT EXISTS idx_publication_vectors_hnsw
ON ai_engine.publication_vectors
USING hnsw (embedding vector_cosine_ops);

-- Tối ưu khi join để xóa sách
CREATE INDEX IF NOT EXISTS idx_publication_vectors_pub_id
ON ai_engine.publication_vectors(publication_id);

-- 3.4 Bảng theo dõi ETL theo hash file (giúp skip xử lý khi file không đổi)
CREATE TABLE IF NOT EXISTS ai_engine.publication_etl_runs (
    publication_id INT PRIMARY KEY REFERENCES library_core.publications(id) ON DELETE CASCADE,
    file_hash VARCHAR(64) NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
