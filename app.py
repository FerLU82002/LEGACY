import streamlit as st
import os
from utils import *
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Asegúrate de que esta sea la primera llamada a Streamlit
st.set_page_config(page_title='preguntaDOC')

# Verifica que el archivo archivos.txt exista
FILE_LIST = "archivos.txt"
if not os.path.exists(FILE_LIST):
    open(FILE_LIST, 'w').close()

st.header("UNHEVAL LEGACY")

# Inicializar la lista de archivos
archivos = load_name_files(FILE_LIST)

upload_password = st.sidebar.text_input("Contraseña para entrenar IA", type="password")

if upload_password == st.secrets["UPLOAD_PASSWORD"]:
    with st.sidebar:
        files_uploaded = st.file_uploader(
            "Carga tu archivo",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button('Procesar'):
            for pdf in files_uploaded:
                if pdf is not None and pdf.name not in archivos:
                    archivos.append(pdf.name)
                    text_to_pinecone(pdf)

            archivos = save_name_files(FILE_LIST, archivos)

    if len(archivos) > 0:
        st.write('Archivos Cargados:')
        archivo_seleccionado = st.selectbox("Selecciona un archivo para borrar", archivos)
        
        delete_password = st.text_input("Ingrese la contraseña para borrar archivos", type="password", key="delete_password")

        if st.button('Borrar Documento Seleccionado'):
            if delete_password == st.secrets["DELETE_PASSWORD"]:
                archivos = [arch for arch in archivos if arch != archivo_seleccionado]
                save_name_files(FILE_LIST, archivos)
                st.experimental_rerun()
            else:
                st.error("Contraseña incorrecta para borrar el archivo")
else:
    st.sidebar.warning("Contraseña incorrecta para subir archivos")

if len(archivos) > 0:
    user_question = st.text_input("Pregunta: ")
    if user_question:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vstore = Pinecone.from_existing_index(INDEX_NAME, embeddings)

        docs = vstore.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=user_question)

        st.write(respuesta)
