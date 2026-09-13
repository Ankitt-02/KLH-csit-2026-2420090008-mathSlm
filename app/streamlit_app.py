import streamlit as st
import sys
import os

# Ensure the root path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.inference import MathSLMInference

st.set_page_config(page_title="MathSLM", page_icon="🧮", layout="centered")

st.title("🧮 MathSLM")
st.markdown("""
Welcome to **MathSLM**, a lightweight decoder-only Transformer built from scratch to solve math word problems.
""")

@st.cache_resource
def load_inference():
    return MathSLMInference()

inferencer = load_inference()

if not inferencer.is_ready():
    st.error("Model or Tokenizer not found! Please run the training pipeline first.")
    st.stop()
    
# Layout
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area("Enter a math word problem:", "If John has 5 apples and gives 2 to Mary, how many apples does John have left?")
    
with col2:
    strategy = st.selectbox("Decoding Strategy", ["greedy", "temperature", "top-k", "self-consistency"])
    
if st.button("Solve", type="primary"):
    with st.spinner(f"Solving using {strategy} decoding..."):
        result = inferencer.solve(question, strategy=strategy)
        
    if "error" in result:
        st.error(result["error"])
    else:
        # Display Results
        st.subheader("Reasoning")
        st.info(result["reasoning"] if result["reasoning"] else "No reasoning generated.")
        
        st.subheader("Final Answer")
        st.success(result["answer"] if result["answer"] else "No answer generated.")
        
        # Verification Badge
        st.subheader("Symbolic Verification")
        if result["verified"]:
            st.markdown("✅ **Verified**: Internal reasoning is mathematically sound.")
        else:
            st.markdown(f"⚠️ **Unverified / Mismatch**: {result['verification_message']}")
            
        # Meta info
        st.caption(f"⏱️ Latency: {result['latency_ms']} ms")
        if strategy == "self-consistency" and "vote_breakdown" in result["meta"]:
            st.caption("🗳️ Vote Breakdown:")
            st.json(result["meta"]["vote_breakdown"])
