#Import required libraries
import os 
from api_key import apiKey, api_Key 

import streamlit as st
import pandas as pd

from langchain.llms import OpenAI, GooglePalm, HuggingFaceHub
from langchain.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool 
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper



#OpenAIKey
os.environ['Google_API_KEY'] = apiKey

os.environ['HUGGINGFACEHUB_API_TOKEN'] = api_Key 
load_dotenv(find_dotenv())

def header(url):
     st.markdown(f'<p style="background-color:#06006C;color:white;font-size:20px;border-radius:2%; padding:2%">{url}</p>', unsafe_allow_html=True)
#Title
st.title('Dataset Analyzer for Lazy Fellows')
 
#Welcoming message
st.write("**Welcome..., My name is Timmy and I am here to mess your with your datasets.**")

#Explanation sidebar 
with st.sidebar:
    st.markdown( """ <style> [data-testid=stSidebar] { background-color:
                #06006C; color: white; } </style> """, unsafe_allow_html=True,)
    st.write('*Your CSV \'s Adventure Begins with Timmy.*')
    st.caption('''**<p style ='color:white'>Timmy was born when an intern (long story short) was too lazy to analyze a dataset his mentor had given. 
               His laziness can be shown through Timmy's performance. He watched only half of the Youtube tutorial to create Timmy. 
               And now he has upload Timmy to Github to get a job. <p style ='color:#85D17B'>If you are a recruiter please....I can do better</p></p>**
    ''',unsafe_allow_html=True)
    st.caption("<p style ='color:white'>Timmy is usnig <b>Google Palm API</b> for this purpose. If you ever feel like forking or cloning this project (First of all: WHY ?), Please use your own API key</p>", unsafe_allow_html=True)

    st.divider()

    st.caption("<p style ='text-align:center'> made by Joel D Joy (not the intern mentioned) </p>",unsafe_allow_html=True )

#Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

#Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        #llm model
        llm = GooglePalm(temperature = 0)
        # llm1 = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature":0.5, "max_length":512,'trust_remote_code':True})
        #Function sidebar
        @st.cache_data
        def steps_eda():
            steps_eda = llm('Write 2 Data Science Jokes')
            return steps_eda

        #Pandas agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)

        #Functions main
        @st.cache_data
        def function_agent():

            

            # Data Overview
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(header(df.head()))

            # Data Cleaning
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(header(columns_df))
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(header(missing_values))
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(header(duplicates))

            # Data Summarization
            st.write("**Data Summarisation**")
            st.write(header(df.describe()))
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(header(correlation_analysis))
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(header(outliers))
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(header(new_features))

            return

        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y =[user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

        @st.cache_resource
        def wiki(prompt):
            wiki_research = WikipediaAPIWrapper().run(prompt)
            return wiki_research

        @st.cache_data
        def prompt_templates():
            data_problem_template = PromptTemplate(
                input_variables=['business_problem'],
                template='Convert the following business problem into a data science problem: {business_problem}.'
            )
            model_selection_template = PromptTemplate(
                input_variables=['data_problem', 'wikipedia_research'],
                template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}, while using this wikipedia research: {wikipedia_research}.'
            )
            return data_problem_template, model_selection_template

        # @st.cache_data
        def chains():
            data_problem_chain = LLMChain(llm=llm, prompt=prompt_templates()[0], verbose=True, output_key='data_problem')
            model_selection_chain = LLMChain(llm=llm, prompt=prompt_templates()[1], verbose=True, output_key='model_selection')
            sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=['business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)
            return sequential_chain

        @st.cache_data
        def chains_output(prompt, wiki_research):
            my_chain = chains()
            my_chain_output = my_chain({'business_problem': prompt, 'wikipedia_research': wiki_research})
            my_data_problem = my_chain_output["data_problem"]
            my_model_selection = my_chain_output["model_selection"]
            return my_data_problem, my_model_selection
        
        @st.cache_data
        def list_to_selectbox(my_model_selection_input):
            algorithm_lines = my_model_selection_input.split('\n')
            algorithms = [algorithm.split(':')[-1].split('.')[-1].strip() for algorithm in algorithm_lines if algorithm.strip()]
            algorithms.insert(0, "Select Algorithm")
            formatted_list_output = [f"{algorithm}" for algorithm in algorithms if algorithm]
            return formatted_list_output
        
        @st.cache_resource
        def python_agent():
            agent_executor = create_python_agent(
                llm=llm,
                tool=PythonREPLTool(),
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
            )
            return agent_executor
        
        @st.cache_data
        def python_solution(my_data_problem, selected_algorithm, user_csv):
            solution = python_agent().run(f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}..Do not use Google Colab. Assume You are using Local machine which does not have google colab. Solve and give me the final answer"
            )
            return solution

        #Main

        st.header('Exploratory data analysis')
        st.subheader('General information about the dataset')

        with st.sidebar:
            with st.expander('Data Science Jokes : Lame'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Variable of study')
        user_question_variable = st.text_input('What variable are you interested in')
        if user_question_variable is not None and user_question_variable !="":
            function_question_variable()

            st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                st.write("")
            
            if user_question_dataframe:
                st.divider()
                st.header("Data Science Problem")
                st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our business problem into a data science problem.")
                
                prompt = st.text_area('What is the business problem you would like to solve?')

                if prompt:                    
                    wiki_research = wiki(prompt)
                    my_data_problem = chains_output(prompt, wiki_research)[0]
                    my_model_selection = chains_output(prompt, wiki_research)[1]
                        
                    st.write(my_data_problem)
                    st.write(my_model_selection)

                    formatted_list = list_to_selectbox(my_model_selection)
                    selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)

                    if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                        st.subheader("Solution")
                        solution = python_solution(my_data_problem, selected_algorithm, user_csv)
                        st.write(solution)

                    
