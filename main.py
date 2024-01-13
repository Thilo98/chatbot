import streamlit as st
import asyncio
from streamlit_chat import message
import PyPDF2
import pickle
import os
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS # FAISS is a vector database
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

import html2text
import requests
import re


st.set_page_config(
    page_title="Easy PDF Reader",
    layout="wide",
    initial_sidebar_state="expanded"
)

#api_key=st.secrets["api_key"]



# since we doont to keep runing frontend code while reciving from openai without intrepuuting while waiting
# to response to come we us async function.
async def main():
    
    async def storeDocEmbeds(file, filename):
        reader = PyPDF2.PdfReader(file)
        corpus = '' . join([p.extract_text() for p in reader.pages if p.extract_text() ])
        
        #Split the document into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(corpus)
        
        #Generate Embeddings of the functions
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        #vectors stores the embeddings corresponidings to each chunk in a file
        vectors = FAISS.from_texts(chunks,embeddings)
         
        #store the vectors in a pickle file
        with open(filename + '.pkl' , "wb") as f:
            pickle.dump(vectors,f)
            
    async def getDocEmbeds(file, filename):
        if not os.path.isfile(filename + '.pkl'):
            await storeDocEmbeds(file,filename)
            
        with open(filename + '.pkl', "rb")as f:
            vectors = pickle.load(f)
            
        return vectors
    
    async def storeStringEmbeds(input_string, filename):
        corpus = input_string
        

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(input_string)

        current_directory = os.getcwd()  # Get the current working directory
        text_loader_kwargs = {'autodetect_encoding': True}
        loader = DirectoryLoader(current_directory, glob="*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        documents = loader.load()
        print('#'*30)
        print("documents: ", documents)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.from_documents(chunks, embeddings)

        with open(filename + ".pkl", "wb") as f:
            pickle.dump(vectors, f)

    async def getStringEmbeds(input_string, filename):
        if not os.path.isfile(filename + ".pkl"):
            await storeStringEmbeds(input_string, filename)

        with open(filename + ".pkl", "rb") as f:
            vectors = pickle.load(f)

        return vectors



    def extract_text_from_url(url):


        response = requests.get(url)
        converter = html2text.HTML2Text()
        converter.ignore_links = True  # Exclude hyperlinks from the extracted text
        text = converter.handle(response.text)

       
        return text
    
    async def conversational_chat(query):

        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
 
        
        
    st.title("PDFChat")
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []
        
    option = st.selectbox("Select Option", ("PDF", "Blog"))
    
    if option == "PDF":
        uploaded_file = st.file_uploader("Choose a file",type="PDF")
        if uploaded_file is not None:
            with st.spinner("Processing...."):
                uploaded_file.seek(0)# This line sets the file cursor position to the beginning of the file (Python File Methods)
                file = uploaded_file.read() #This line reads the contents of the file from the current cursor position until the end of the file and stores the data in the variable named
                vectors = await getDocEmbeds(io.BytesIO(file),uploaded_file.name)
                
                #retreive previous converation and pass it to the next prompt
                qa=ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo",
                                                                    openai_api_key=api_key), #returns source documnets from the prevous statements allow and tell which chunks were used from these vectors that from questions
                                                         retriever=vectors.as_retriever(),
                                                        return_source_documents=True)
    
            st.session_state["ready"] =True
            
    elif option == "Blog":
        url = st.text_input("Enter the URL of the blog")
        
        if url:
            with st.spinner("Processing..."):
                content = extract_text_from_url(url)
                print(content)
                print('#'*30)
                valid_file_name = re.sub(r'\W+', '', url)
                vectors = await getStringEmbeds(content, f"{valid_file_name}.txt")
                print('#'*30)
                qa=ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo",
                                                                    openai_api_key=api_key), #returns source documnets from the prevous statements allow and tell which chunks were used from these vectors that from questions
                                                         retriever=vectors.as_retriever(),
                                                        return_source_documents=True)
                print('#'*30)
  
            st.session_state["ready"] = True
        
    # elif option == "Database":
    #     uploaded_file = st.file_uploader("Choose a file",type="db")
    #     if uploaded_file is not None:
    #         with st.spinner("Processing...."):
    #             uploaded_file.seek(0)# This line sets the file cursor position to the beginning of the file (Python File Methods)
    #             file = uploaded_file.read() #This line reads the contents of the file from the current cursor position until the end of the file and stores the data in the variable named
                
    
    #         st.session_state["ready"] =True  
            
    if st.session_state.get('ready' , False): 
        
        # The get() method returns the value of the item with the specified key. 
        # --> dictionary.get(keyname, value) 
        # --> keyname - (Required)The keyname of the item you want to return the value from 
        # --> value - (Optional) A value to return if the specified key does not exist. Default value None 
        # --> In here If there is no value in "ready" it returns as 'False'(bool) but this function work if there is value --> True 
        # --> ---Refer Python Dictionary Methods---
        
        
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcome to Thilo's PDF chatbot! You can talk with your pdf now"]
            
        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey"]
        
        container = st.container() 
        response_container = st.container()
        
        with container:
            with st.form(key='my_form',clear_on_submit=True):
                user_input = st.text_input("Problem", placeholder="e.g: Summarize the document")
                # Every form must have a submit button.
                submit_button = st.form_submit_button(label='Submit')
                
                if submit_button and user_input:
                    output = await conversational_chat(user_input)
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)
                    
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    if i < len(st.session_state['past']):
                        st.markdown(
                            "<div style='background-color: #90caf9; color: black; padding: 10px; border-radius: 5px; width: 70%; float: right; margin: 5px;'>"+ st.session_state["past"][i] +"</div>",
                            unsafe_allow_html=True
                        )
                    # st.markdown(
                    #     "<div style='background-color: #c5e1a5; color: black; padding: 10px; border-radius: 5px; width: 70%; float: left; margin: 5px;'>"+ st.session_state["generated"][i] +"</div>",
                    #     unsafe_allow_html=True
                    # )
                    # message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
                
            
                
                
                                                
        
if __name__=="__main__":
    asyncio.run(main())
