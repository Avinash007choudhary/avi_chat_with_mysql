from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import plotly.express as px
import pandas as pd
import json
import re

# Initialize database (unchanged)
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)



# Modified to potentially return visualization instructions
def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# New function to create visualizations
def create_visualization(data, chart_type="auto"):
    try:
        # Convert to DataFrame if not already
        df = pd.DataFrame(data)
        
        # Handle empty data
        if df.empty:
            return None
            
        # Auto-detect chart type
        if chart_type == "auto":
            if len(df.columns) == 1:
                # Single column - histogram
                chart_type = "histogram"
            elif len(df.columns) == 2:
                if pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                    if len(df) > 10:  # Many data points - line may be better
                        chart_type = "line"
                    elif pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                        chart_type = "line"
                    else:
                        chart_type = "bar"
                else:
                    chart_type = "pie"
            else:
                # For 3+ columns, use first two numeric columns
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if len(numeric_cols) >= 2:
                    chart_type = "scatter"
                else:
                    chart_type = "bar"
        
        # Create appropriate visualization
        if chart_type == "bar":
            fig = px.bar(df, 
                        x=df.columns[0], 
                        y=df.columns[1],
                        title=f"Bar Chart: {df.columns[1]} by {df.columns[0]}")
                        
        elif chart_type == "line":
            fig = px.line(df,
                         x=df.columns[0],
                         y=df.columns[1],
                         title=f"Trend of {df.columns[1]} over {df.columns[0]}")
                         
        elif chart_type == "pie":
            fig = px.pie(df,
                         names=df.columns[0],
                         values=df.columns[1],
                         title=f"Distribution of {df.columns[1]} by {df.columns[0]}")
                         
        elif chart_type == "histogram":
            fig = px.histogram(df,
                              x=df.columns[0],
                              title=f"Distribution of {df.columns[0]}")
                              
        elif chart_type == "scatter":
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            fig = px.scatter(df,
                           x=numeric_cols[0],
                           y=numeric_cols[1],
                           title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
                           
        else:
            return None
            
        # Improve layout
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="closest"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

# Modified response function to handle visualizations
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema, question, sql query, and sql response, write a natural language response.
    When the user asks for data that would be better visualized (like comparisons, distributions, or trends), 
    include a visualization instruction like this: [VISUALIZATION:chart_type] where chart_type can be bar, line, or pie.
    
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# Main app (modified to handle visualizations)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")
st.title("Chat with MySQL")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    host = st.text_input("Host", value="localhost", key="Host")
    port = st.text_input("Port", value="3306", key="Port")
    user = st.text_input("User", value="root", key="User")
    password = st.text_input("Password", type="password", value="admin", key="Password")
    database = st.text_input("Database", value="Chinook", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(user, password, host, port, database)
                st.session_state.db = db
                st.success("Connected to database!")
            except Exception as e:
                st.error(f"Failed to connect: {e}")

# Display chat history with potential visualizations
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            # Check if the message contains a visualization instruction
            if hasattr(message, 'visualization_data'):
                st.markdown(message.content)
                fig = create_visualization(message.visualization_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Handle new user query
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        if 'db' not in st.session_state:
            st.error("Please connect to the database first")
        else:
            # Get the text response
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)


            # Check if response contains visualization instruction
            vis_match = re.search(r'\[VISUALIZATION:(\w+)\]', response)
            if vis_match:
                chart_type = vis_match.group(1)
                response = response.replace(vis_match.group(0), "")
                
                # Get the data for visualization by running the query again
                try:
                    sql_chain = get_sql_chain(st.session_state.db)
                    query = sql_chain.invoke({
                        "question": user_query,
                        "chat_history": st.session_state.chat_history
                    })
                    data = st.session_state.db.run(query)
                    
                    # Convert SQL result to DataFrame
                    if data:
                        # Create a custom AIMessage that includes visualization data
                        vis_message = AIMessage(content=response)
                        vis_message.visualization_data = json.loads(data)
                        st.session_state.chat_history.append(vis_message)
                        
                        # Display both text and visualization
                        st.markdown(response)
                        fig = create_visualization(vis_message.visualization_data, chart_type)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to generate visualization: {e}")
                    st.markdown(response)
                    st.session_state.chat_history.append(AIMessage(content=response))
            else:
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))