import os
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms import Together
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings

from agent import get_wikipedia_summary

# Carga las variables de entorno desde el archivo .env
load_dotenv()
# Accede a la API key utilizando os.environ
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

model = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0,
    max_tokens=1024,
    top_k=20,
    together_api_key=TOGETHER_API_KEY
)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


load_vector_store = Chroma(
    persist_directory="stores/ConserGPT/", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})


# Provide a template following the LLM's original chat template.
template = """ Utiliza la siguiente información para responder a la pregunta del usuario.
Si no sabes la respuesta, di simplemente que no la sabes, no intentes inventarte una respuesta ya que si te inventas algo, puedes cometer errores.

Contexto: {context}
Pregunta: {question}

La vida de muchas personas depende de que tu respuesta sea correcta.
Devuelve sólo la respuesta útil que aparece a continuación y nada más.
Responde solo y exclusivamente con la información que se te ha sido proporcionada.
Responde siempre en castellano.
Solo y exclusivamente si el usuario te pide buscar en wikipedia, ejecuta el siguiente código: {SearchInWiki}; en el caso contrario, omite este paso y no lo ejecutes.
Respuesta útil:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough(
    ), "SearchInWiki": RunnableLambda(get_wikipedia_summary)}
    | prompt
    | model
    | StrOutputParser()
)


def get_response(input):
    query = input
    output = chain.invoke(query)

    return output


input = gr.Text(
    label="Prompt",
    show_label=False,
    max_lines=1,
    placeholder="Enter your prompt",
    container=False,
)

iface = gr.Interface(fn=get_response,
                     inputs=input,
                     outputs="text",
                     title="ConserGPT",
                     description="This is a RAG implementation based on Mixtral.",
                     allow_flagging='never'
                     )

iface.launch(share=True)
