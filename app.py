import streamlit as st
from hybrid_router import load_vectorstore, answer

st.title("Manufacturing Assistant")


@st.cache_resource
def get_store():
    return load_vectorstore()


vectorstore = get_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            route, text, docs = answer(vectorstore, question)
        st.markdown(text)
        st.caption(f"Route: {route}")
        if docs:
            with st.expander("Sources"):
                for d in docs:
                    st.write(f"{d.metadata['source']} (chunk {d.metadata['chunk_id']})")

    st.session_state.messages.append({"role": "assistant", "content": text})