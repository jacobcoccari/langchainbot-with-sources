from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()



from utls.compose_prompt import create_prompt
from utls.format_memory import get_chat_history
from utls.moderation import harmful_content_check
# from utls.token_printer import TokenPrinter


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def generate_assistant_response(query, retriever, streamlit_memory):
    history = get_chat_history(streamlit_memory)
    prompt = create_prompt(history) 
    model = ChatOpenAI(model = 'gpt-4-1106-preview', temperature=0)
    check = harmful_content_check(query)
    # callback = TokenPrinter()
    if check is not None:
        print(check)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
    )
    return chain.invoke(query).content  
