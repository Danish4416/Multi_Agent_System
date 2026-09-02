import streamlit as st
from pipeline import run_research_pipeline

st.set_page_config(
    page_title="Multi-Agent AI Research System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent AI Research System")

st.write(
    "Search, analyze and generate detailed research reports using AI agents."
)

topic = st.text_input(
    "Enter your research topic",
    placeholder="e.g. Impact of AI on financial fraud detection"
)

if st.button("🚀 Generate Research Report", type="primary"):

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        status = st.status(
            "🤖 Research agents are working...",
            expanded=True
        )

        try:

            def update_progress(message):
                status.write(message)

            result = run_research_pipeline(
                topic,
                progress_callback=update_progress
            )

            status.update(
                label="✅ Research completed!",
                state="complete",
                expanded=False
            )

            st.divider()

            st.header("📄 Final Research Report")

            st.markdown(result["report"])

        except Exception as e:

            status.update(
                label="❌ Research failed",
                state="error"
            )

            st.error(f"Something went wrong: {e}")