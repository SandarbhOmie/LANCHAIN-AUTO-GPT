
'''
Shri Ram hanuman
Langchain: Spiderman with jet packs
** it provides the ability to use agents

Their are 6 key modules in Langchain:
1. MOdels:  gives us access us to large language models
2. Prompts : prompts structure our prompts with Templates
3. Idexes: prepare our documents to work with large language models
4. Memory : Gives us access to Llm chain access the historial inputs
5. Chain: to string it all together
6. Agents: tools like wikipedia and google search

Open Ai : APi Key== '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'

Here we are installing following dependencies
1. Streamlit : USed to BUild The APP

2. LANGCHAIN: used to build the LLM workflow

3. OPENAI : Needed to use Open AI

4. Wikipedia : used to connect gpt to wikipedia

5. chromadb : vector storage.

6. TIKTOKEN : backend Tokenizer for Openai


using a prompt template simplifies our stuff.


USES OF CHAIN HERE:
task1 >> task2 
'''

import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


# it allow us to run topic from prompt template and then go and generate output
from langchain.chains import LLMChain, SequentialChain

# for memory block
from langchain.memory import ConversationBufferMemory

from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY']=apikey

# app Screen
st.title('🦜️🔗 SANDOM GPT')
prompt = st.text_input('Plug in the PROMPT TO create the script')


# prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a youtube video title about {topic}'
)

# script 
script_template = PromptTemplate(
    input_variables=['title','wikipedia_research'],
    template='write me a youtube video script about this title {title} while leveraging this wikipedia reaerch:{wikipedia_research}'
)

# memory
title_memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title',memory_key='chat_history')


# Llms : creating a intance of Open ai
llm = OpenAI(temperature = 0.9)
title_chain = LLMChain(llm=llm,prompt =title_template,verbose=True,output_key='title',memory=title_memory)
script_chain = LLMChain(llm=llm,prompt =script_template,verbose=True,output_key='script',memory=script_memory)
#sequential_chain=SequentialChain(chains=[title_chain, script_chain],input_variables=['topic'],output_variables=['title','script'],verbose=True)

wiki = WikipediaAPIWrapper()
# trigerring the prompt
# shows stuffs to the prompt
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title,wikipedia_research=wiki_research)
    #response = sequential_chain({'topic':prompt})

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)
    with st.expander('Wikipedia History'):
        st.info(wiki_research)