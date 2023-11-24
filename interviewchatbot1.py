import streamlit as st
from dataclasses import dataclass
from typing import Literal
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
from langchain.callbacks import get_openai_callback

# Custom CSS for improved visual design
custom_css = """
<style>
    .chat-bubble {
        padding: 10px;
        margin: 5px;
        border-radius: 10px;
        max-width: 70%;
    }
    .human-bubble {
        background-color: #333;  /* Dark gray background for human messages on black background */
        color: #FFF;  /* White text color for human messages */
        float: left;
    }
    .ai-bubble {
        background-color: #333;  /* Dark gray background for AI messages on black background */
        color: #FFF;  /* White text color for AI messages */
        float: right;
    }
    .button-container {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar contents
with st.sidebar:
    st.title('🤖 Interview Chatbot by LangChain 💼')
    st.write("Welcome to the Interview Chatbot powered by LangChain and OpenAI!")
    st.write("About this Chatbot:")
    st.markdown(
        "This chatbot simulates an interview experience. "
        "You can ask the bot to conduct an interview for you. "
        "The bot will engage you in a conversation by asking domain-specific interview questions one at a time. "
        "At the end, it will provide feedback on how the complete interview went for you."
    )
    st.markdown("[LangChain](https://python.langchain.com/)")
    st.markdown("[OpenAI](https://platform.openai.com/docs/models)")
    st.markdown("[Streamlit](https://streamlit.io/)")
    st.write("Made by [Sampik Kumar Gupta]")

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        initialize_conversation()

def prompt_template():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""As an interviewer conducting a brief interview, your goal is to comprehensively perform the role of an interviewer. 
        Begin by requesting the user to introduce themselves.Subsequently, inquire about the domain or field in which they intend to undergo the interview.
        Proceed to engage the user in conversation by posing domain-specific interview questions.
        Throughout the conversation, Pose only a single question at a time. Refrain from providing answers independently to any questions during the interview
        and Conclude the interview by offering feedback on the overall experience for the user.
        """),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

def initialize_conversation():
    llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(return_messages=True),
        prompt=prompt_template(),
    )

def on_click_callback():
    #Get the OpenAI callback
    with get_openai_callback() as cb:
        #Retrieve the user's input prompt from the Streamlit session state
        human_prompt = st.session_state.human_prompt

        #Retrieve the chat history and LangChain conversation from the Streamlit session state
        history = st.session_state.history
        conversation = st.session_state.conversation
        
        #Use LangChain to generate an AI response based on the user's input prompt
        llm_response = conversation.predict(input=human_prompt)
        
        #Append the user's input and AI response to the chat history
        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", llm_response))
        
        #Update the token count in the Streamlit session state
        st.session_state.token_count += cb.total_tokens

def display_chat_input():
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="Hello Interview Bot",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )

def display_chat_history():
    for chat in st.session_state.history:
        if chat.origin == "human":
            bubble_class = "human-bubble"
            label = "You👤"
        elif chat.origin == "ai":
            bubble_class = "ai-bubble"
            label = "Interviewer🤖"
        else:
            bubble_class = ""
            label = "Unknown"

        div = f"""
        <div class="chat-bubble {bubble_class}">
            <strong>{label}:</strong> &#8203;{chat.message}
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

    for _ in range(3):
        st.markdown("")

def run():
    exit_chat = st.button("Exit Interview")
    new_chat = st.button("Start New Interview")

    if exit_chat:
        st.session_state.history = []
        st.session_state.token_count = 0
        st.session_state.conversation.memory = ConversationBufferMemory(return_messages=True)
        st.write("You exited from the interview. You can start a new Interview by clicking the New Interview Button.")
        
    if new_chat:
        initialize_conversation()

    with st.container():
        display_chat_history()

    with st.form(key="unique_chat_form_key"):
        display_chat_input()

    st.caption(f"""
    Used {st.session_state.token_count} tokens \n
    Debug Langchain conversation: 
    {st.session_state.conversation.memory.buffer}
    """)

# main function
def main():
    load_dotenv()
    st.title("🤖 Interview Buddy: Your Virtual Interview Practice Partner 💼")
    initialize_session_state()
    run()

# Run the application
if __name__ == "__main__":
    main()

