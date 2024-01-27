import os
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# Obtén la ruta completa del directorio actual del script
script_directory = os.path.dirname(os.path.abspath(__file__))
md_folder_path = os.path.join(script_directory, "md_folder")

for filename in os.listdir(md_folder_path):
    try:
        # Construye la ruta completa del archivo
        file_path = os.path.join(md_folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as archivo:
            contenido = archivo.read()
            print(f"Se leyó el archivo '{file_path}'.")

            headersToSplitOn = [("#", "Header"), ("##", "Title")]

            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headersToSplitOn)
            md_header_splits = markdown_splitter.split_text(contenido)

            for document in md_header_splits:
                lista = []

                # Extraer y mostrar los metadatos
                metadata = document.metadata
                page_content = document.page_content
                for key, value in metadata.items():
                    lista.append(f"{value}{page_content}")

            vector_store = Chroma.from_documents(md_header_splits, embeddings, collection_metadata={
                                                 "hnsw:space": "cosine"}, persist_directory="stores/ConserGPT")

    except FileNotFoundError:
        print(f"El archivo '{file_path}' no se encontró.")
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo: {e}")
