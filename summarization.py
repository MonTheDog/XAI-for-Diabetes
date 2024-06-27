from openai import OpenAI
import streamlit as st

def ask_gpt():
    client = OpenAI(
        api_key=""
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Say this is a test!"
            }
        ],
        model="gpt-3.5-turbo",
    )

    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    prompt = st.chat_input("Say something")
    if prompt:
        st.write("Test")