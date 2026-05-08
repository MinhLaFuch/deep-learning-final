"""
VQA Comparison App — API Version
Gọi 4 model VQA qua REST API endpoint.

Cài đặt:
    pip install streamlit requests Pillow

Chạy:
    streamlit run vqa_app_api.py
"""

import streamlit as st
import requests
import base64
import time
from PIL import Image
from io import BytesIO

# ─────────────────────────────────────────────
# ✏️  CẤU HÌNH MODEL — điền thông tin API vào đây
# ─────────────────────────────────────────────
MODELS = {
    "Model 1": {
        "api_url": "http://localhost:8001/predict",   # ← thay bằng endpoint thực
        "api_key": "",                                # ← thay bằng API key (nếu có)
        "description": "Mô tả model 1",
    },
    "Model 2": {
        "api_url": "http://localhost:8002/predict",
        "api_key": "",
        "description": "Mô tả model 2",
    },
    "Model 3": {
        "api_url": "http://localhost:8003/predict",
        "api_key": "",
        "description": "Mô tả model 3",
    },
    "Model 4": {
        "api_url": "http://localhost:8004/predict",
        "api_key": "",
        "description": "Mô tả model 4",
    },
}

TIMEOUT = 30  # giây


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_model_api(model_name: str, image: Image.Image, question: str) -> dict:
    """
    ✏️  CHỈNH SỬA HÀM NÀY theo định dạng request của từng API.
    Hiện tại gửi JSON với 2 field: image (base64) và question.
    Trả về dict: {"answer": str, "confidence": float (optional), "latency": float}
    """
    cfg = MODELS[model_name]
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    payload = {
        "image": image_to_base64(image),   # ← chỉnh theo API spec
        "question": question,              # ← chỉnh theo API spec
    }

    t0 = time.time()
    response = requests.post(
        cfg["api_url"],
        json=payload,
        headers=headers,
        timeout=TIMEOUT,
    )
    latency = time.time() - t0
    response.raise_for_status()

    data = response.json()

    return {
        "answer": data.get("answer", "N/A"),          # ← chỉnh key theo response thực
        "confidence": data.get("confidence", None),   # ← optional
        "latency": round(latency, 3),
    }


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
def render_result_card(model_name: str, result: dict | None, error: str | None):
    with st.container(border=True):
        st.markdown(f"**{model_name}**")
        st.caption(MODELS[model_name]["description"])
        if error:
            st.error(f"❌ {error}")
        elif result:
            st.success(f"💬 {result['answer']}")
            cols = st.columns(2)
            cols[0].metric("Latency", f"{result['latency']}s")
            if result["confidence"] is not None:
                cols[1].metric("Confidence", f"{result['confidence']:.2%}")
        else:
            st.info("Chưa có kết quả")


def main():
    st.set_page_config(page_title="VQA Comparison — API", layout="wide")
    st.title("🔍 VQA Model Comparison")
    st.caption("Phiên bản: **API**  |  Upload ảnh, nhập câu hỏi, so sánh 4 model cùng lúc")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        timeout_val = st.slider("Timeout (giây)", 5, 120, TIMEOUT)
        run_parallel = st.checkbox("Gọi song song (threading)", value=True)
        st.divider()
        st.subheader("API Endpoints")
        for name, cfg in MODELS.items():
            st.text_input(f"{name} URL", value=cfg["api_url"], key=f"url_{name}")

    # ── Input ──
    col_img, col_q = st.columns([1, 1])
    with col_img:
        uploaded = st.file_uploader("📷 Upload ảnh", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)
    with col_q:
        question = st.text_area("❓ Câu hỏi", placeholder="Ví dụ: What color is the car?", height=120)
        run_btn = st.button("🚀 Chạy tất cả model", type="primary", use_container_width=True)

    st.divider()

    # ── Kết quả ──
    if "results" not in st.session_state:
        st.session_state.results = {name: None for name in MODELS}
        st.session_state.errors = {name: None for name in MODELS}

    if run_btn:
        if not uploaded:
            st.warning("Vui lòng upload ảnh.")
        elif not question.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            st.session_state.results = {name: None for name in MODELS}
            st.session_state.errors = {name: None for name in MODELS}

            if run_parallel:
                import concurrent.futures
                def fetch(name):
                    # dùng URL từ sidebar nếu đã chỉnh
                    MODELS[name]["api_url"] = st.session_state.get(f"url_{name}", MODELS[name]["api_url"])
                    try:
                        return name, call_model_api(name, image, question), None
                    except Exception as e:
                        return name, None, str(e)

                with st.spinner("Đang gọi API..."):
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(fetch, n) for n in MODELS]
                        for f in concurrent.futures.as_completed(futures):
                            name, res, err = f.result()
                            st.session_state.results[name] = res
                            st.session_state.errors[name] = err
            else:
                for name in MODELS:
                    with st.spinner(f"Đang gọi {name}..."):
                        try:
                            st.session_state.results[name] = call_model_api(name, image, question)
                        except Exception as e:
                            st.session_state.errors[name] = str(e)

    # ── Render cards ──
    cols = st.columns(4)
    for col, name in zip(cols, MODELS):
        with col:
            render_result_card(
                name,
                st.session_state.results.get(name),
                st.session_state.errors.get(name),
            )


if __name__ == "__main__":
    main()