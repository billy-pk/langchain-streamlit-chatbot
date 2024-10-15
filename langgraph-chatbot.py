# import the required libraries
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()


# set the environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_PROJECT"] = "YOUR-PROJECT-NAME"
os.environ['LANGSMITH_API_KEY']= os.environ.get("LANGSMITH_API_KEY")
os.environ['TAVILY_API_KEY'] = os.environ.get('TAVILY_API_KEY')
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')

# create a model instance
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0, max_retries= 1,n=1)


class MessagesState(MessagesState):
    pass # because messages key is builtin 


tool = TavilySearchResults(max_results=1) # tavilySearch is a tool provided by langchain to search web
tools = [tool]

llm_with_tools = llm.bind_tools(tools) # bind llm with tool

sys_msg = SystemMessage(content = " You are a helpful assistant.u will reply briefly")
def assistant(state:MessagesState):
    return {'messages' : [llm_with_tools.invoke([sys_msg] + state['messages'])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools",ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant",tools_condition)
builder.add_edge("tools", "assistant")
# compile the graph with memory
memory = MemorySaver()
graph = builder.compile(checkpointer = memory)

# Initialize session state for conversation history if not already done
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []


def invoke_graph(user_input):
    
    # Add the new user input to the conversation history
    st.session_state["conversation_history"].append(HumanMessage(content=str(user_input)))
    config = {'configurable' : {'thread_id' : '1'}}
    #messages = [HumanMessage(content = str(user_input))]
    #response = graph.invoke({"messages" : messages}, config)

    response = graph.invoke({"messages": st.session_state["conversation_history"]}, config)
    
    # Get the AI's response and add it to the conversation history
    ai_message = response['messages'][-1]
    st.session_state["conversation_history"].append(ai_message)
    return ai_message.content


st.set_page_config(page_title= 'Langgraph ChatBot', page_icon= "random", layout= 'wide' )

st.title("Langgraph ChatBot using Tavily Search")


# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

if "quit" not in st.session_state:
    st.session_state["quit"] = False

if "goodbye_shown" not in st.session_state:
    st.session_state["goodbye_shown"] = False

# If quit flag is set but goodbye message has not been shown, display it
if st.session_state["quit"] and not st.session_state["goodbye_shown"]:
    st.write("Exiting the application. Goodbye!")
    st.session_state["goodbye_shown"] = True  # Mark goodbye message as shown
    st.experimental_rerun()  # Rerun the app to exit after message is shown

# If quit flag is set and goodbye has been shown, exit the app
if st.session_state["quit"] and st.session_state["goodbye_shown"]:
    os._exit(0)  # Terminate the app completely. It also stops the streamlit server

# Get user input if the user has not quit yet
user_input = st.chat_input("Please enter your query (type 'quit' to exit):")

if user_input:
    # If the user inputs "quit", mark it in session state and rerun
    if user_input.lower() == "quit":
        st.session_state["quit"] = True
        st.experimental_rerun()  # Rerun to handle exit logic

    # Otherwise, process the input by invoking the graph
    result = invoke_graph(user_input)
    st.session_state["conversation"].append((user_input, result))

    # Display the conversation history

    for user_query, response in st.session_state["conversation"]:
        st.write(f"**You:** {user_query}")
        st.write(f"**Assistant:** {response}")


    # # Display only the latest conversation (last user query and assistant response)
    # latest_query, latest_response = st.session_state["conversation"][-1]
    
    # st.write(f"**You:** {latest_query}")
    # st.write(f"**Assistant:** {latest_response}")

