import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# TODO: Hoàn thiện hàm này — đọc file product_catalog.json, lọc theo category và max_price.
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    with open(catalog_file, "r", encoding="utf-8") as f:
        products = json.load(f)

    if max_price is None:
        max_price = 999999999999

    category = (category or "").strip().lower()
    results = [
        p for p in products
        if str(p.get("category", "")).lower() == category
        and p.get("price_vnd", 0) <= max_price
    ]
    return results


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# TODO: Hoàn thiện hàm này — tạo ticket mới và lưu vào support_tickets.json.
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    priority = (priority or "medium").strip().lower()
    if priority not in ("low", "medium", "high"):
        priority = "medium"

    # Load existing tickets
    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as f:
            try:
                existing_tickets = json.load(f)
            except json.JSONDecodeError:
                existing_tickets = []

    # Generate ticket ID: sequence = (max existing seq) + 1 để tránh trùng ID
    today = datetime.now().strftime("%Y%m%d")
    max_seq = 0
    for t in existing_tickets:
        try:
            max_seq = max(max_seq, int(str(t.get("ticket_id", "")).split("-")[-1]))
        except ValueError:
            continue
    seq = max(max_seq, len(existing_tickets)) + 1
    ticket_id = f"TK-{today}-{seq:03d}"

    # Create & save new ticket
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat() + "+07:00",
        "category": "general",
    }
    existing_tickets.append(new_ticket)

    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(existing_tickets, f, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "priority": priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# TODO: Định nghĩa JSON Schema cho từng tool (name, description, parameters).
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, tour/khách sạn Vinpearl) "
            "theo danh mục và giá tối đa. Dùng khi khách hỏi về sản phẩm, giá, gợi ý mua/đặt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' (xe điện) hoặc 'du_lich' (du lịch/nghỉ dưỡng).",
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ (ví dụ 600 triệu = 600000000). Bỏ qua nếu không giới hạn.",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Tạo ticket hỗ trợ khi khách hàng báo lỗi, khiếu nại, yêu cầu bảo hành "
            "hoặc cần nhân viên hỗ trợ xử lý."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên khách hàng.",
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả chi tiết vấn đề khách hàng gặp phải.",
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên: 'low', 'medium' hoặc 'high'. Mặc định 'medium'.",
                    "enum": ["low", "medium", "high"],
                },
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
