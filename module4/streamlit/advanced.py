from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st


st.set_page_config(page_title="LangChain Agents + MRKL", page_icon="🦜")
st.title("🦜 LangChain Agents + MRKL")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant",
         "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Will Donald Trump be indicted?"):
    st.session_state.messages.append({"role": "user",
                                      "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    
    llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        streaming = True
    )

    search_agent = initialize_agent(
        tools = load_tools(["ddg-search"]),
        llm = llm,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors = True
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        st.write(response)