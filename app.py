import streamlit as st

from src.config import BASE_DIR
from src.rag_engine import ask_rag

st.set_page_config(
    page_title="The Batch — Multimodal AI Search",
    layout="wide",
)

st.title("The Batch — Multimodal RAG Explorer")
st.markdown(
    "Ask questions about AI news from **DeepLearning.AI's The Batch**. "
    "The system retrieves relevant articles and **visual media (charts, diagrams)** "
    "using ChromaDB + Gemini."
)

with st.sidebar:
    st.header("Retrieval Settings")
    top_k_text = st.slider("Text excerpts to retrieve", min_value=1, max_value=8, value=4)
    top_k_images = st.slider("Max images to retrieve", min_value=0, max_value=5, value=3)
    min_img_sim = st.slider("Min image similarity", min_value=0.0, max_value=0.9, value=0.50, step=0.05)

    st.divider()
    st.subheader("Example Questions")
    examples = [
        "How does DeepSeek-R1 compare to OpenAI o1 on benchmarks and cost?",
        "What is DeepLearning.AI Pro and what does it include?",
        "What new AI hardware or chips were discussed recently?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query_input"] = ex
    
query = st.text_input(
    "Enter your question:",
    key="query_input",
    placeholder="e.g., How does DeepSeek-R1 perform on AIME 2024 and MATH-500?"
)

search_clicked = st.button("Search & Answer", type="primary")

if search_clicked and query.strip():
    with st.spinner("Searching articles and analyzing visual media..."):
        try:
            result = ask_rag(
                query=query.strip(),
                top_k_text=top_k_text,
                top_k_images=top_k_images,
                min_image_similarity=min_img_sim,
            )
        except Exception as ex:
            st.error(f"An error occurred: {ex}")
            st.stop()

    col_answer, col_media = st.columns([3, 2], gap="large")

    with col_answer:
        st.subheader("AI Answer")
        st.markdown(result["answer"])

        st.divider()
        st.subheader("Source Articles")
        if result["sources"]:
            for src in result["sources"]:
                st.markdown(f"- [{src['title']}]({src['url']})")
        else:
            st.info("No matching source articles found.")

        with st.expander("Inspect Retrieved Text Excerpts"):
            for idx, txt in enumerate(result["retrieved_texts"], start=1):
                st.markdown(
                    f"**Excerpt {idx}** — *{txt['metadata']['title']}* "
                    f"(Similarity: `{txt['similarity']}`)"
                )
                st.caption(txt["content"])
                st.divider()

    with col_media:
        st.subheader("Retrieved Media")
        if result["retrieved_images"]:
            for img_hit in result["retrieved_images"]:
                rel_path = img_hit["metadata"].get("image_path", "").replace("\\", "/")
                full_path = BASE_DIR / rel_path
                if full_path.exists():
                    st.image(
                        str(full_path),
                        caption=(
                            f"{img_hit['metadata']['title']} "
                            f"(Similarity: {img_hit['similarity']})"
                        ),
                        use_container_width=True,
                    )
        else:
            st.info("No images met the similarity threshold for this query.")
elif search_clicked:
    st.info("Please enter a question first.")