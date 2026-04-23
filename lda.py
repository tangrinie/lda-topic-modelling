import streamlit as st

st.set_page_config(page_title="My Simple Page", page_icon="📄")

st.title("My Streamlit Page")

st.markdown("""
# Hello

This is a **simple Streamlit page**.

## What it shows
- A title
- Markdown text
- A button

You can write:
- **bold**
- *italic*
- `code`

[Streamlit website](https://streamlit.io)
""")

if st.button("Click me"):
    st.success("Button clicked!")
