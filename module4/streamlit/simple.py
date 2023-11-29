import os
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

load_dotenv()

# initialize llm
llm = OpenAI(temperature=0, streaming=True)

# intialize DuckDuckGo search tools
tools = load_tools(['ddg-search'])
# creating agent that connect llm to tools
agent = initialize_agent(
    tools=tools, llm = llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Create input, if input provided then assign to prompt
if prompt := st.chat_input():
    # display user question and assistant icon
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st.write("🧠 Thinking..")
        # create StreamlitCallbackHandler as container
        st_callback = StreamlitCallbackHandler(st.container())
        # run agent to answer the prompt
        response = agent.run(prompt, callbacks = [st_callback])
        # write response
        st.write(response)