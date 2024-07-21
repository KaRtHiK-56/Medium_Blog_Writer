import streamlit as st 
from langchain.llms import Ollama 
from langchain.prompts import PromptTemplate
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.chains.router import MultiPromptChain
from langchain.chains import LLMChain
from langchain.chains import create_retrieval_chain

st.title("Medium Blog Writer 🤖")
blog  = st.text_area("Tell me a topic to write a blog on:")
words = st.text_area("Within how many words:",height=50)
styles = st.selectbox("Select the style of your blog",("Simple",
"Narrative",
"Technical",
"Technological",
"Creative",))


def blogger(blog,words,styles):
    simple=''' Write a blog for {styles} style for a topic {blog}
        within {words} words.The blog should be in simple plain and in understanding context  '''
    
    narative=''' Write a blog for {styles} style for a topic {blog}
        within {words} words.The blog should be in a narrative way as a teacher a grandma narates a stories. '''
    
    technical='''  Write a blog for {styles} style for a topic {blog}
        within {words} words.The blog should be way more technical, explaining the core components and strong technical point of view '''
    
    technological=''' Write a blog for {styles} style for a topic {blog}
        within {words} words.The blog should be explaining the technological concepts and its impacts  '''
    
    creative=''' Write a blog for {styles} style for a topic {blog}
        within {words} words.The blog should be more creative but keeping the {blog} in context and more easy to understand  '''

    prompt_infos =[
        { "name":"simple" ,
         "description":"creating blogs according to simple style" ,
         "template":simple, },

         { "name":"narrative" ,
         "description":"creating blogs according to narative style" ,
         "template": narative, },

         { "name":"technical" ,
         "description":"creating blogs according to technical style" ,
         "template":technical, },

         { "name":"technological" ,
         "description":"creating blogs according to technological style" ,
         "template": technological, },

         { "name":"creative" ,
         "description":"creating blogs according to creative style" ,
         "template": creative, },
    ]

    llm = Ollama(model='llama3',temperature=0.7)

    destination_p = {}
    
    for info in prompt_infos:
        name = info['name']
        prompt_template = info['template']
        prompt = PromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm = llm,prompt = prompt)
        destination_p['name'] = chain

    default_prompt = PromptTemplate.from_template(template='{input}')
    default_chain = LLMChain(llm=llm,prompt=default_prompt)

    destination = [f"{p['name']}:{p['description']}" for p in prompt_infos] 
    destination_str = "/n".join(destination)

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destination_str)

    router_prompt = PromptTemplate(template=router_template,input_variables=['blog','words','styles'],output_parser=RouterOutputParser())

    router_chain = LLMRouterChain.from_llm(llm,router_prompt)

    chain = MultiPromptChain(router_chain=router_chain,destination_chains=destination_p,default_chain=default_chain,verbose=True,silent_errors=True)


    chains = create_retrieval_chain(router_chain,chain)
    answer = chains.invoke({"input":{blog,words,styles}})
    print(answer)
    return answer['answer']['text']


submit = st.button("Generate")
if submit:
    with st.spinner("Generating ..."):
        st.write(blogger(blog,words,styles))