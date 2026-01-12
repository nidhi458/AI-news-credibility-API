import streamlit as st
import requests

st.set_page_config(page_title="AI News Credibility Checker", page_icon="📰")
st.title("📰 AI News Credibility Checker")

# Input fields
title = st.text_input("News Title:")
content = st.text_area("News Content:", height=200)

# Use the correct FastAPI route
api_url = "https://ai-news-credibility-api.onrender.com/analyze"

if st.button("Check News"):
    if not title.strip() or not content.strip():
        st.warning("❗ Please enter both title and content.")
    else:
        try:
            response = requests.post(
                api_url,
                json={"title": title, "content": content}
            )
            data = response.json()
            st.success("✅ Prediction complete!")
            st.write(f"**Label:** {data['prediction']}")
            st.write(f"**Confidence:** {data['confidence']}")
        except Exception as e:
            st.error(f"⚠️ Error calling API: {e}")
