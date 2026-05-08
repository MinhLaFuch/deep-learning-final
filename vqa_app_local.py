"""
VQA Comparison App — Local Model Version
Load 4 model VQA trực tiếp từ local (PyTorch / HuggingFace).

Cài đặt:
    pip install streamlit torch transformers Pillow

Chạy:
    streamlit run vqa_app_local.py
"""

import time
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# ✏️  INSERT MODEL CỦA BẠN VÀO ĐÂY
# Mỗi model cần implement 2 thứ:
#   1. load_fn()  → trả về object model (được cache)
#   2. infer_fn(model, image, question) → trả về (answer: str, confidence: float | None)
# ─────────────────────────────────────────────

# ── Model 1 ──────────────────────────────────
def _load_model_1():
    """✏️ Load model 1 ở đây (ví dụ: HuggingFace pipeline, torch.load, v.v.)"""
    # Ví dụ HuggingFace:
    # from transformers import pipeline
    # return pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
    raise NotImplementedError("Chưa insert model 1")


def _infer_model_1(model, image: Image.Image, question: str) -> tuple[str, float | None]:
    """✏️ Inference model 1, trả về (answer, confidence hoặc None)"""
    # Ví dụ HuggingFace pipeline:
    # result = model(image, question, top_k=1)
    # return result[0]["answer"], result[0]["score"]
    raise NotImplementedError("Chưa insert inference model 1")


# ── Model 2 ──────────────────────────────────
def _load_model_2():
    """✏️ Load model 2"""
    raise NotImplementedError("Chưa insert model 2")


def _infer_model_2(model, image: Image.Image, question: str) -> tuple[str, float | None]:
    """✏️ Inference model 2"""
    raise NotImplementedError("Chưa insert inference model 2")


# ── Model 3 ──────────────────────────────────
def _load_model_3():
    """✏️ Load model 3"""
    raise NotImplementedError("Chưa insert model 3")


def _infer_model_3(model, image: Image.Image, question: str) -> tuple[str, float | None]:
    """✏️ Inference model 3"""
    raise NotImplementedError("Chưa insert inference model 3")


# ── Model 4 ──────────────────────────────────
def _load_model_4():
    """✏️ Load model 4"""
    raise NotImplementedError("Chưa insert model 4")


def _infer_model_4(model, image: Image.Image, question: str) -> tuple[str, float | None]:
    """✏️ Inference model 4"""
    raise NotImplementedError("Chưa insert inference model 4")


# ─────────────────────────────────────────────
# Registry — map tên model → (load_fn, infer_fn, mô tả)
# ─────────────────────────────────────────────
MODELS = {
    "Model 1": {
        "load_fn": _load_model_1,
        "infer_fn": _infer_model_1,
        "description": "Mô tả model 1",
        "device": "cuda",   # hoặc "cpu"
    },
    "Model 2": {
        "load_fn": _load_model_2,
        "infer_fn": _infer_model_2,
        "description": "Mô tả model 2",
        "device": "cuda",
    },
    "Model 3": {
        "load_fn": _load_model_3,
        "infer_fn": _infer_model_3,
        "description": "Mô tả model 3",
        "device": "cuda",
    },
    "Model 4": {
        "load_fn": _load_model_4,
        "infer_fn": _infer_model_4,
        "description": "Mô tả model 4",
        "device": "cuda",
    },
}


# ─────────────────────────────────────────────
# Cache model với st.cache_resource (load 1 lần duy nhất)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(name: str):
    cfg = MODELS[name]
    return cfg["load_fn"]()


def run_inference(name: str, image: Image.Image, question: str) -> dict:
    model = get_model(name)
    t0 = time.time()
    answer, confidence = MODELS[name]["infer_fn"](model, image, question)
    latency = round(time.time() - t0, 3)
    return {"answer": answer, "confidence": confidence, "latency": latency}


# ─────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────
def render_result_card(name: str, result: dict | None, error: str | None, loading: bool):
    with st.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(MODELS[name]["description"])

        if loading:
            with st.spinner("Đang inference..."):
                pass  # spinner hiện ở ngoài
        elif error:
            st.error(f"❌ {error}")
        elif result:
            st.success(f"💬 {result['answer']}")
            cols = st.columns(2)
            cols[0].metric("Latency", f"{result['latency']}s")
            if result["confidence"] is not None:
                cols[1].metric("Confidence", f"{result['confidence']:.2%}")
        else:
            st.info("Chưa có kết quả")


def check_models_loaded() -> dict[str, bool]:
    status = {}
    for name in MODELS:
        try:
            get_model(name)
            status[name] = True
        except NotImplementedError:
            status[name] = False
        except Exception:
            status[name] = False
    return status


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    st.set_page_config(page_title="VQA Comparison — Local", layout="wide")
    st.title("🔍 VQA Model Comparison")
    st.caption("Phiên bản: **Local**  |  Upload ảnh, nhập câu hỏi, so sánh 4 model cùng lúc")

    # ── Sidebar: trạng thái model ──
    with st.sidebar:
        st.header("📦 Trạng thái Model")
        if st.button("🔄 Kiểm tra / Load models"):
            status = check_models_loaded()
            for name, ok in status.items():
                if ok:
                    st.success(f"✅ {name} sẵn sàng")
                else:
                    st.warning(f"⚠️ {name} chưa có model")

        st.divider()
        st.header("⚙️ Tuỳ chọn")
        show_confidence = st.checkbox("Hiển thị Confidence", value=True)
        device_info = st.empty()

        try:
            import torch
            device = "🟢 CUDA" if torch.cuda.is_available() else "🟡 CPU"
        except ImportError:
            device = "⚠️ torch chưa cài"
        device_info.info(f"Device: {device}")

    # ── Input ──
    col_img, col_q = st.columns([1, 1])
    with col_img:
        uploaded = st.file_uploader("📷 Upload ảnh", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)
    with col_q:
        question = st.text_area("❓ Câu hỏi", placeholder="Ví dụ: How many people are in the image?", height=120)
        run_btn = st.button("🚀 Chạy tất cả model", type="primary", use_container_width=True)

    st.divider()

    # ── Session state ──
    if "results" not in st.session_state:
        st.session_state.results = {n: None for n in MODELS}
        st.session_state.errors = {n: None for n in MODELS}

    # ── Inference ──
    if run_btn:
        if not uploaded:
            st.warning("Vui lòng upload ảnh.")
        elif not question.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            st.session_state.results = {n: None for n in MODELS}
            st.session_state.errors = {n: None for n in MODELS}

            prog = st.progress(0, text="Đang chạy inference...")
            for i, name in enumerate(MODELS):
                prog.progress((i) / len(MODELS), text=f"Đang chạy {name}...")
                try:
                    st.session_state.results[name] = run_inference(name, image, question)
                except NotImplementedError:
                    st.session_state.errors[name] = "Model chưa được insert. Vui lòng điền vào _load/_infer functions."
                except Exception as e:
                    st.session_state.errors[name] = str(e)
            prog.progress(1.0, text="Hoàn tất!")

    # ── Render result cards ──
    cols = st.columns(4)
    for col, name in zip(cols, MODELS):
        with col:
            render_result_card(
                name,
                st.session_state.results.get(name),
                st.session_state.errors.get(name),
                loading=False,
            )

    # ── So sánh nhanh (nếu có đủ kết quả) ──
    completed = {n: r for n, r in st.session_state.results.items() if r is not None}
    if len(completed) > 1:
        st.divider()
        st.subheader("📊 So sánh nhanh")
        import pandas as pd
        rows = []
        for name, res in completed.items():
            rows.append({
                "Model": name,
                "Answer": res["answer"],
                "Latency (s)": res["latency"],
                "Confidence": f"{res['confidence']:.2%}" if res["confidence"] is not None else "N/A",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Biểu đồ latency
        st.bar_chart(df.set_index("Model")["Latency (s)"])


if __name__ == "__main__":
    main()