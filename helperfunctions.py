import os
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
import azure.cognitiveservices.speech as speechsdk
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI


import json
from langchain.prompts import PromptTemplate

import json
import shutil
import streamlit as st

import logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

from typing import List
from langchain import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import MultiQueryRetriever

# refresh vector index list from directory names in faiss folder
def refresh_vector_index_list():
    print(st.session_state.project)
    directory_list = list()
    for root, dirs, files in os.walk("projects/"+st.session_state.project+"/faiss/", topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(name))
    st.session_state.vector_index_list = directory_list


def resetpage():
    if 'startpage' in st.session_state:
        del st.session_state.startpage
    st.session_state.startpage=1
    if 'endpage' in st.session_state:
        del st.session_state.endpage
    st.session_state.endpage=len(st.session_state.pagecontent)

# refresh vector index list from directory names in faiss folder
def refresh_topic_list():
    directory_list = list()
    for root, dirs, files in os.walk("projects/"+st.session_state.project+"/topics/", topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(name))
    st.session_state.topic_list = directory_list

def add_topic(topicname):
    print('adding topic to '+st.session_state.project)
    if os.path.isdir("projects/"+st.session_state.project+"/topics/"+topicname):
        st.error("Topic already exists")
    else:
        os.mkdir("projects/"+st.session_state.project+"/topics/"+topicname)
        #write questions.txt to dir
        with open("projects/"+st.session_state.project+"/topics/"+topicname+"/questions.txt", "w") as f:
            pass
        #write queries.txt to dir
        with open("projects/"+st.session_state.project+"/topics/"+topicname+"/queries.txt", "w") as f:
            pass
        #write ground_truth.txt to dir
        with open("projects/"+st.session_state.project+"/topics/"+topicname+"/ground_truth.txt", "w") as f:
            pass
    refresh_topic_list()
    st.session_state.topic=topicname
    load_topic()  


def delete_topic(topicname):
    if os.path.isdir("projects/"+st.session_state.project+"/topics/"+topicname):
        #delete directory
       shutil.rmtree("projects/"+st.session_state.project+"/topics/"+topicname, ignore_errors=True)
    refresh_topic_list()
    if(len(st.session_state.topic_list)>0):
        st.session_state.topic=st.session_state.topic_list[0]
        load_topic()     
        
def refresh_project_list():
    if 'project_list' in st.session_state:
        del st.session_state.project_list
    dirs = [entry.path for entry in os.scandir('projects') if entry.is_dir()]
    #replace "projects\\" with " 
    for i in range(len(dirs)):
        dirs[i]=dirs[i].replace("\\","/")
        dirs[i]=dirs[i].replace("projects/","")
    st.session_state.project_list = dirs
    if(len(dirs)>0):
        st.session_state.project=dirs[0]

def add_project(projectname):
    if os.path.isdir("projects/"+projectname):
        st.error("project already exists")
    else:
        os.mkdir("projects/"+projectname)
        #write questions.txt to dir
        os.mkdir("projects/"+projectname+"/files")
        os.mkdir("projects/"+projectname+"/faiss")
        os.mkdir("projects/"+projectname+"/topics")
    refresh_project_list()
    st.session_state.project=projectname
    loadproject()



def delete_project(projectname):
    if os.path.isdir("projects/"+projectname):
        #delete directory
       shutil.rmtree("projects/"+projectname, ignore_errors=True)
    refresh_project_list()
    if(len(st.session_state.project_list)>0):
        st.session_state.project=st.session_state.project_list[0]
        loadproject()
        

# refresh vector index list from directory names in faiss folder
# def refresh_vector_index_list(): 
#     directory_list = list()
#     for root, dirs, files in os.walk("projects/"+st.session_state.project+"/faiss/", topdown=False):
#         for name in dirs:
#             directory_list.append(os.path.join(name))
#     st.session_state.vector_index_list = directory_list

def setquestion():
    if(st.session_state['question']!="-"):
        st.session_state['questioninput']=st.session_state['question']

def getmessages(systemessage,question):
    from langchain.schema import(
    HumanMessage,
    SystemMessage
    )
    messages = [
    SystemMessage(content=systemessage),
    HumanMessage(content=question)
    ]
    return messages
    
# Speech to Text with Azure Cognitive Services
def recognize_from_microphone(target):
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION" which are loaded from the .env file in main
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    speech_config.speech_recognition_language=st.session_state.language

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    st.info("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        if target=="question":
            add_question(speech_recognition_result.text)      
            st.session_state.question=speech_recognition_result.text
        else:
            add_query(speech_recognition_result.text)   
            st.session_state.query=speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

def askquestion():
    ans = askwithcontext(st.session_state.question)
    st.session_state.answer=ans


def gettokens(text):
    import tiktoken
    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    return len(enc.encode(text))

# Text to Speech with Azure Cognitive Services
# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
def synthesize_text(text):
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    # The language of the voice that speaks.
    
    if st.session_state.language=="de-DE":
        speech_config.speech_synthesis_voice_name='de-DE-KatjaNeural'
    else:
        speech_config.speech_synthesis_voice_name='en-US-JennyNeural'
    
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text")
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")

def load_embeddings():
    print("loading embeddings")
    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=16)
    # find all folders in faiss folder
    directory_list = list()
    for root, dirs, files in os.walk("projects/"+st.session_state.project+"/faiss/", topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(name))
    st.session_state.vector_index_list = directory_list
    # load vector store from selected folder
    st.session_state.vs = list()
    st.session_state.document_name = list()
    for root, dirs, files in os.walk("projects/"+st.session_state.project+"/faiss/", topdown=False):
        for name in files:
            if name.endswith(".faiss"):
                print(root)
                st.session_state.vs.append(FAISS.load_local(root, embeddings))
                st.session_state.document_name.append(root)
                contentjsonfile = root.replace("faiss","files") + ".pagecontent.json"
                # if contentjsonfile exists, load it
                if os.path.isfile(contentjsonfile):
                    with open(contentjsonfile, encoding='utf-8') as json_file:
                        st.session_state.pagecontent = json.load(json_file)
                else:
                    logging.debug('No Pagecontent '+ name+' found.')
                tablemdfile = root.replace("faiss","files") + ".tables.md"
                # if tablemdfile exists, load it
                if os.path.isfile(tablemdfile):
                    with open(tablemdfile, encoding='utf-8') as table_file:
                        st.session_state.tables = table_file.read()
                else:    
                    logging.debug('No Tables from '+ name+' found.')
                fullmdfile = root.replace("faiss","files") +".md"
                # if fullmdfile exists, load it
                if os.path.isfile(fullmdfile):
                    with open(fullmdfile, encoding='utf-8') as fullmd_file:
                        st.session_state.fullmd = fullmd_file.read()
                else:
                    logging.debug('No Full MD from '+ name+' found.')
                keyvaluesjsonfile = root.replace("faiss","files") +".keyvalues.json"
                # if keyvaluesjsonfile exists, load it
                if os.path.isfile(keyvaluesjsonfile):
                    with open(keyvaluesjsonfile, encoding='utf-8') as json_file:
                        st.session_state.keyvalues = json.load(json_file)
                else:
                    logging.debug('No Key Values from '+ name+' found.')
    # check if st.session_state.startpage is not inisantiated yet
    if 'startpage' not in st.session_state:
        st.session_state.startpage=1
    resetpage()

def getgroundtruthpages():
    groundtruthpages = []
    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/ground_truth.txt') as f:
        for line in f:
            if line.split(";")[0] == st.session_state.vector_index_name:
                groundtruthpages=line.split(";")[1].split(",")
                for i in range(len(groundtruthpages)):
                    groundtruthpages[i]=int(groundtruthpages[i])
            #print(groundtruthpages)
    return groundtruthpages


def setgroundtruthpages():
    newgroundtruthpages = st.session_state.ground_truth
    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/ground_truth.txt','r') as f:
        lines = f.readlines()

    addline=True
    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/ground_truth.txt','w') as f:
        for line in lines:
            if line.split(";")[0] == st.session_state.vector_index_name:
                addline=False
                if newgroundtruthpages=="\n":
                    f.write(st.session_state.vector_index_name+";"+newgroundtruthpages+"\n")
            else:
                f.write(line)
    if addline:
        with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/ground_truth.txt','a') as f:
            f.write(st.session_state.vector_index_name+";"+newgroundtruthpages+"\n")
    
# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


def multi_query_retriver(question, vs, llm, n_version, important_note=""):

    output_parser = LineListOutputParser()

    if important_note == "":
        prompt = f"You are an AI language model assistant. Your task is to generate {n_version} \n" + \
        """different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions seperated by newlines.

            ---------

            """ + \
            f"Important Note: {important_note}\n" + \
            """---------
            Original question: {question}"""
    else:
        prompt = f"You are an AI language model assistant. Your task is to generate {n_version}" + \
        """different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions seperated by newlines. \n """ + \
            """Original question: {question}"""

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template= prompt,
    )

    # Chain
    llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

    retrievers = [
        MultiQueryRetriever(retriever=v.as_retriever(), llm_chain=llm_chain, parser_key="lines")  # "lines" is the key (attribute name) of the parsed output 
        for v in vs
        ]
    document_lists = [r.get_relevant_documents(query=question) for r in retrievers]
    documents = [doc for doc_list in document_lists for doc in doc_list]
    print(f"documents pages: {' '.join([str(doc.metadata['pages'][0]) for doc in documents])}")
    return documents

       
def askwithcontext(question):
    load_embeddings()
    vs = st.session_state.vs
    important_note = "Compensation for credit risk is deemed an acceptable component of interest. Adjustments to this compensation based on the credit risk development of an asset do not necessarily jeopardize the SPPI characteristic."
    q_nversion = 5
    llm=AzureChatOpenAI(deployment_name='gpt-35-turbo', temperature=0.0)
    documents = multi_query_retriver(question=question, vs=vs, 
                                     llm=llm, n_version=q_nversion, 
                                     important_note=important_note)
    
    llm4=AzureChatOpenAI(deployment_name='gpt-4', temperature=0.0)
    chain = load_qa_chain(llm4, chain_type="map_rerank", return_intermediate_steps=True)
    answer=chain({"input_documents": documents, "question": question, "important_note":important_note}, return_only_outputs=False)
        
    return answer

def delete_query():
    queryname=st.session_state.query
    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/queries.txt', 'r') as f:
        lines = f.readlines()

    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/queries.txt', 'w') as f:
        for line in lines:
            if line.strip("\n") != queryname:
                f.write(line)
                #print(line)
    load_topic(False)
    if 'context' in st.session_state:
        del st.session_state.context

                                
def add_query(queryname):
    #add query to end of queries.txt
    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/queries.txt', 'a') as f:
        f.write(queryname+"\n")
    load_topic(False)

def delete_question():
    questionname=st.session_state.question
    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/questions.txt', 'r') as f:
        lines = f.readlines()

    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/questions.txt', 'w') as f:
        for line in lines:
            if line.strip("\n") != questionname:
                f.write(line)
                #print(line)
    load_topic(False)
    if 'answer' in st.session_state:
        del st.session_state.answer

                                
def add_question(questionname):
    #add question to end of questions.txt
    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/questions.txt', 'a') as f:
        f.write(questionname+"\n")
    load_topic(False)


def load_topic(reset=True):
    if reset:
        if 'context' in st.session_state:
            del st.session_state.context
        if 'answer' in st.session_state:
            del st.session_state.answer
    #open queries text and add all lines to a list
    query_list = []
    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/queries.txt') as f:
        for line in f:
            query_list.append(line.strip())
    st.session_state.query_list = query_list
    #open questions text and add all lines to a list
    question_list = []
    with open("projects/"+st.session_state.project+'/topics/'+st.session_state.topic+'/questions.txt') as f:
        for line in f:
            question_list.append(line.strip())
    st.session_state.question_list = question_list

    
    
def loadproject():
    if 'context' in st.session_state:
        del st.session_state.context
    if 'answer' in st.session_state:
        del st.session_state.answer
    if 'vector_index_list' in st.session_state:
        del st.session_state.vector_index_list
    if 'topic_list' in st.session_state:
        del st.session_state.topic_list
    if 'vector_index_name' in st.session_state:
        del st.session_state.vector_index_name
    if 'topic_list' in st.session_state:
        del st.session_state.topic_list
    refresh_vector_index_list()
    refresh_topic_list()
    if len(st.session_state.topic_list)>0:
        st.session_state.topic=st.session_state.topic_list[0]
        load_topic()
        #load_embeddings()

    
    
