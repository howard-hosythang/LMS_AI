from __future__ import annotations

import re
import unicodedata


GENERIC_TAGS = {
    "ai",
    "book",
    "chapter",
    "content",
    "course",
    "document",
    "education",
    "engineering",
    "example",
    "figure",
    "introduction",
    "knowledge",
    "learning",
    "numbers",
    "publication",
    "result",
    "science",
    "student",
    "study",
    "system",
    "table",
    "technology",
    "textbook",
    "variables",
    "ấn phẩm",
    "bài học",
    "chương",
    "công nghệ",
    "giáo dục",
    "giáo trình",
    "học tập",
    "khoa học",
    "kỹ thuật",
    "nội dung",
    "sách",
    "sinh viên",
    "tài liệu",
    "tri thức",
}

CATALOG_LABELS = {
    "title",
    "subtitle",
    "description",
    "authors",
    "categories",
    "publisher",
}

TAG_ALIASES = {
    "machine learning": "Học Máy",
    "deep learning": "Học Sâu",
    "supervised learning": "Học Có Giám Sát",
    "feature engineering": "Kỹ Thuật Đặc Trưng",
    "model validation": "Đánh Giá Mô Hình",
    "artificial intelligence": "Trí Tuệ Nhân Tạo",
    "data science": "Khoa Học Dữ Liệu",
    "database": "Cơ Sở Dữ Liệu",
    "databases": "Cơ Sở Dữ Liệu",
    "normalization": "Chuẩn Hóa Dữ Liệu",
    "query processing": "Xử Lý Truy Vấn",
    "transaction": "Giao Dịch Dữ Liệu",
    "big data": "Dữ Liệu Lớn",
    "algorithm": "Thuật Toán",
    "algorithms": "Thuật Toán",
    "data structures": "Cấu Trúc Dữ Liệu",
    "web development": "Lập Trình Web",
    "software architecture": "Kiến Trúc Phần Mềm",
    "clean code": "Mã Sạch",
    "domain driven design": "Thiết Kế Hướng Miền",
    "software testing": "Kiểm Thử Phần Mềm",
    "operating systems": "Hệ Điều Hành",
    "computer networks": "Mạng Máy Tính",
    "cybersecurity": "Bảo Mật Thông Tin",
    "distributed systems": "Hệ Phân Tán",
    "cloud computing": "Điện Toán Đám Mây",
    "compiler design": "Trình Biên Dịch",
    "computer architecture": "Kiến Trúc Máy Tính",
    "discrete mathematics": "Toán Rời Rạc",
    "probability statistics": "Xác Suất Thống Kê",
    "linear algebra": "Đại Số Tuyến Tính",
    "optimization": "Tối Ưu Hóa",
    "electronics": "Mạch Điện Tử",
    "embedded systems": "Hệ Thống Nhúng",
    "control systems": "Điều Khiển Tự Động",
    "mechanical design": "Thiết Kế Cơ Khí",
    "chemical engineering": "Kỹ Thuật Hóa Học",
    "environment": "Môi Trường",
    "materials science": "Công Nghệ Vật Liệu",
    "logistics": "Hậu Cần",
    "finance": "Tài Chính",
    "research methods": "Nghiên Cứu Khoa Học",
}

VI_TAG_TRANSLATIONS = {
    "Học Máy": "Machine Learning",
    "Học Sâu": "Deep Learning",
    "Học Có Giám Sát": "Supervised Learning",
    "Kỹ Thuật Đặc Trưng": "Feature Engineering",
    "Đánh Giá Mô Hình": "Model Validation",
    "Trí Tuệ Nhân Tạo": "Artificial Intelligence",
    "Khoa Học Dữ Liệu": "Data Science",
    "Cơ Sở Dữ Liệu": "Database",
    "Chuẩn Hóa Dữ Liệu": "Database Normalization",
    "Xử Lý Truy Vấn": "Query Processing",
    "Giao Dịch Dữ Liệu": "Database Transactions",
    "Dữ Liệu Lớn": "Big Data",
    "Thuật Toán": "Algorithms",
    "Cấu Trúc Dữ Liệu": "Data Structures",
    "Lập Trình Python": "Python Programming",
    "Lập Trình Java": "Java Programming",
    "Lập Trình JavaScript": "JavaScript Programming",
    "Lập Trình Web": "Web Development",
    "Kiến Trúc Phần Mềm": "Software Architecture",
    "Mã Sạch": "Clean Code",
    "Thiết Kế Hướng Miền": "Domain Driven Design",
    "Kiểm Thử Phần Mềm": "Software Testing",
    "Hệ Điều Hành": "Operating Systems",
    "Mạng Máy Tính": "Computer Networks",
    "Bảo Mật Thông Tin": "Cybersecurity",
    "Hệ Phân Tán": "Distributed Systems",
    "Điện Toán Đám Mây": "Cloud Computing",
    "Trình Biên Dịch": "Compiler Design",
    "Kiến Trúc Máy Tính": "Computer Architecture",
    "Toán Rời Rạc": "Discrete Mathematics",
    "Xác Suất Thống Kê": "Probability And Statistics",
    "Đại Số Tuyến Tính": "Linear Algebra",
    "Tối Ưu Hóa": "Optimization",
    "Mạch Điện Tử": "Electronic Circuits",
    "Hệ Thống Nhúng": "Embedded Systems",
    "Điều Khiển Tự Động": "Control Systems",
    "Cơ Học Kỹ Thuật": "Engineering Mechanics",
    "Thiết Kế Cơ Khí": "Mechanical Design",
    "Robot": "Robotics",
    "Kết Cấu Xây Dựng": "Structural Engineering",
    "Kỹ Thuật Hóa Học": "Chemical Engineering",
    "Môi Trường": "Environment",
    "Công Nghệ Vật Liệu": "Materials Science",
    "Hậu Cần": "Logistics",
    "Tài Chính": "Finance",
    "Marketing": "Marketing",
    "Nghiên Cứu Khoa Học": "Research Methods",
    "Kiến Thức Nền Tảng": "Foundational Knowledge",
    "Phương Pháp Thực Hành": "Practical Methods",
    "Ứng Dụng Thực Tế": "Practical Applications",
    "Bài Toán Chuyên Ngành": "Domain Problems",
    "Tư Duy Phân Tích": "Analytical Thinking",
    "Nội Dung Chuyên Ngành": "Domain Topics",
    "Tư Duy Kỹ Thuật": "Engineering Thinking",
}

FALLBACK_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Học Có Giám Sát", ("supervised learning", "labelled examples", "classification", "regression")),
    ("Kỹ Thuật Đặc Trưng", ("feature engineering", "feature selection", "feature extraction")),
    ("Đánh Giá Mô Hình", ("model validation", "generalization", "overfitting", "error analysis")),
    ("Học Máy", ("machine learning", "học máy", "supervised", "classification", "regression")),
    ("Học Sâu", ("deep learning", "học sâu", "neural network", "cnn", "rnn", "transformer")),
    ("Trí Tuệ Nhân Tạo", ("artificial intelligence", "trí tuệ nhân tạo", "heuristic", "planning")),
    ("Khoa Học Dữ Liệu", ("data science", "khoa học dữ liệu", "dữ liệu", "data mining", "analytics")),
    ("SQL", ("sql", "query language")),
    ("Chuẩn Hóa Dữ Liệu", ("normalization", "database normalization", "chuẩn hóa dữ liệu")),
    ("Xử Lý Truy Vấn", ("query processing", "query optimization", "truy vấn")),
    ("Giao Dịch Dữ Liệu", ("transaction", "concurrency control", "recovery")),
    ("Cơ Sở Dữ Liệu", ("database", "cơ sở dữ liệu", "relational")),
    ("Dữ Liệu Lớn", ("big data", "hadoop", "spark", "distributed data")),
    ("Thuật Toán", ("algorithm", "thuật toán", "complexity", "dynamic programming")),
    ("Cấu Trúc Dữ Liệu", ("data structure", "cấu trúc dữ liệu", "tree", "graph", "hash table")),
    ("Lập Trình Python", ("python", "django", "flask", "pandas", "numpy")),
    ("Lập Trình Java", ("java", "jvm", "spring", "oop", "object oriented")),
    ("Lập Trình JavaScript", ("javascript", "typescript", "node.js", "react")),
    ("Lập Trình Web", ("web development", "html", "css", "frontend", "backend")),
    ("Kiến Trúc Phần Mềm", ("software architecture", "design pattern", "microservice")),
    ("Mã Sạch", ("clean code", "refactoring", "code smell", "unit test")),
    ("Thiết Kế Hướng Miền", ("domain driven design", "ddd", "bounded context")),
    ("Kiểm Thử Phần Mềm", ("software testing", "test case", "unit testing")),
    ("Hệ Điều Hành", ("operating system", "linux", "process", "thread", "memory management")),
    ("Mạng Máy Tính", ("computer network", "tcp", "ip", "routing", "http")),
    ("Bảo Mật Thông Tin", ("security", "cybersecurity", "cryptography", "authentication")),
    ("Hệ Phân Tán", ("distributed system", "replication", "consensus", "fault tolerance")),
    ("Điện Toán Đám Mây", ("cloud computing", "aws", "azure", "virtualization")),
    ("Trình Biên Dịch", ("compiler", "lexer", "parser", "code generation")),
    ("Kiến Trúc Máy Tính", ("computer architecture", "processor", "cache", "instruction set")),
    ("Toán Rời Rạc", ("discrete mathematics", "logic", "graph theory", "combinatorics")),
    ("Xác Suất Thống Kê", ("probability", "statistics", "random variable")),
    ("Đại Số Tuyến Tính", ("linear algebra", "matrix", "vector", "eigenvalue")),
    ("Tối Ưu Hóa", ("optimization", "linear programming", "convex", "gradient")),
    ("Mạch Điện Tử", ("electronics", "circuit", "transistor", "amplifier")),
    ("Hệ Thống Nhúng", ("embedded", "microcontroller", "firmware", "sensor")),
    ("Điều Khiển Tự Động", ("control system", "pid", "feedback", "stability")),
    ("Cơ Học Kỹ Thuật", ("mechanics", "statics", "dynamics", "stress", "strain")),
    ("Thiết Kế Cơ Khí", ("mechanical design", "machine design", "manufacturing")),
    ("Robot", ("robot", "robotics", "kinematics", "automation")),
    ("Kết Cấu Xây Dựng", ("structural", "concrete", "steel structure", "load")),
    ("Kỹ Thuật Hóa Học", ("chemical engineering", "reactor", "separation process")),
    ("Môi Trường", ("environment", "pollution", "wastewater", "sustainability")),
    ("Công Nghệ Vật Liệu", ("materials science", "polymer", "composite", "semiconductor")),
    ("Hậu Cần", ("logistics", "supply chain", "inventory", "transportation")),
    ("Tài Chính", ("finance", "investment", "portfolio", "capital")),
    ("Marketing", ("marketing", "brand", "consumer behavior", "market research")),
    ("Nghiên Cứu Khoa Học", ("research method", "methodology", "literature review")),
)


def normalize_key(value: str) -> str:
    normalized = unicodedata.normalize("NFD", str(value or "").lower())
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("đ", "d")
    without_marks = re.sub(r"[^a-z0-9+#.]+", " ", without_marks)
    return " ".join(without_marks.split())


def clean_ai_tag(value: str) -> str:
    tag = re.sub(r"[_\s]+", " ", str(value or "")).strip()
    tag = re.sub(r"[^\wÀ-ỹ+#.\- ]+", "", tag, flags=re.UNICODE)
    tag = re.sub(r"\s+", " ", tag).strip(" -_.")
    tag = re.sub(r"(?i)^(tag|keyword|từ khóa)\s*[:\-]\s*", "", tag).strip()
    if not tag:
        return ""
    alias = TAG_ALIASES.get(normalize_key(tag))
    if alias:
        return alias
    if tag.islower():
        tag = tag.title()
    return tag[:50].strip(" -_.")


def parse_catalog_context(publication_context: str) -> dict[str, str]:
    context: dict[str, str] = {}
    for line in publication_context.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in CATALOG_LABELS and value:
            context[key] = value
    return context


def is_low_quality_tag(tag: str, publication_context: str = "") -> bool:
    clean = clean_ai_tag(tag)
    if not clean or len(clean) < 3 or len(clean) > 50:
        return True

    key = normalize_key(clean)
    if key in {normalize_key(item) for item in GENERIC_TAGS}:
        return True

    words = clean.split()
    if len(words) > 5:
        return True

    context = parse_catalog_context(publication_context)
    blocked_values = [
        context.get("title", ""),
        context.get("subtitle", ""),
        context.get("authors", ""),
    ]
    if any(key and key == normalize_key(value) for value in blocked_values):
        return True

    categories = [
        part.strip()
        for part in re.split(r"[,;/]", context.get("categories", ""))
        if part.strip()
    ]
    if any(key == normalize_key(category) for category in categories):
        return True

    return False


def dedupe_quality_tags(tags: list[str], publication_context: str = "", limit: int = 5) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = clean_ai_tag(raw_tag)
        key = normalize_key(tag)
        if not tag or key in seen or is_low_quality_tag(tag, publication_context):
            continue
        result.append(tag)
        seen.add(key)
        if len(result) >= limit:
            break
    return result


def english_tag_for_vi(tag: str) -> str:
    clean = clean_ai_tag(tag)
    key = normalize_key(clean)
    for vi_tag, en_tag in VI_TAG_TRANSLATIONS.items():
        if key == normalize_key(vi_tag):
            return en_tag
    return clean


def align_english_tags(tags_vi: list[str], tags_en: list[str] | None = None) -> list[str]:
    aligned: list[str] = []
    seen: set[str] = set()
    tags_en = tags_en or []

    for index, tag_vi in enumerate(tags_vi):
        candidate = str(tags_en[index]).strip() if index < len(tags_en) else ""
        if not candidate:
            candidate = english_tag_for_vi(tag_vi)
        candidate = re.sub(r"\s+", " ", candidate).strip(" -_.")[:100]
        if not candidate:
            candidate = english_tag_for_vi(tag_vi)
        key = normalize_key(candidate)
        if key in seen:
            candidate = english_tag_for_vi(tag_vi)
            key = normalize_key(candidate)
        if key not in seen:
            aligned.append(candidate)
            seen.add(key)

    return aligned[: len(tags_vi)]


def suggest_tags_from_text(text: str, publication_context: str = "", limit: int = 5) -> list[str]:
    normalized = f" {normalize_key(text)} "
    normalized_plain = normalized.replace(".", " ")
    scored: list[tuple[int, int, str]] = []
    for index, (tag, keywords) in enumerate(FALLBACK_CONCEPTS):
        score = 0
        for keyword in (tag, *keywords):
            key = normalize_key(keyword)
            key_plain = key.replace(".", " ")
            if key and (f" {key} " in normalized or f" {key_plain} " in normalized_plain):
                score += 3 if normalize_key(tag) == key else 1
        if score:
            scored.append((-score, index, tag))

    scored.sort()
    candidates = [tag for _score, _index, tag in scored]
    return dedupe_quality_tags(candidates, publication_context, limit)
