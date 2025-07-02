from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import plotly.express as px
import pandas as pd
import re
from decimal import Decimal

# Initialize database
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# SQL Generation Chain
def get_sql_chain(db):
    template = """You are a SQL expert. Given this schema:
    <SCHEMA>{schema}</SCHEMA>
    
    Generate ONLY the executable SQL query for: {question}
    - No explanations
    - No markdown formatting
    - No leading/trailing text
    - Must begin with SELECT/INSERT/UPDATE/DELETE
    
    Bad Examples:
    "Here is the SQL: SELECT ..."
    ```sql
    SELECT ... 
    ```
    
    Good Example:
    SELECT * FROM table WHERE condition;
    
    Question: {question}
    SQL:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    
    def validate_sql(sql: str):
        """Ensure we only get executable SQL"""
        sql = sql.strip()
        if not re.match(r'^\s*(SELECT|INSERT|UPDATE|DELETE|WITH)', sql, re.IGNORECASE):
            raise ValueError("Generated query doesn't start with valid SQL keyword")
        return sql.split(';')[0]  # Take only the first statement
    
    return (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info())
        | prompt
        | llm 
        | StrOutputParser()
        | validate_sql
    )

# Data Processing
def parse_sql_data(data):
    """Handle multiple SQL result formats"""
    if isinstance(data, pd.DataFrame):
        return data
    
    if isinstance(data, str):
        try:
            # Handle tuple strings with Decimals
            if data.startswith("[(") and "Decimal" in data:
                pattern = r"\('([^']+)', Decimal\('([\d.]+)'\)\)"
                matches = re.findall(pattern, data)
                return pd.DataFrame(matches, columns=['category', 'value'])
            return pd.DataFrame([x.strip() for x in data.split('\n') if x.strip()])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame(data)

# Visualization Engine
def create_visualization(data, chart_type="auto"):
    try:
        df = parse_sql_data(data)
        if df.empty or len(df) < 1:
            return None

        # Auto-detect chart type
        if chart_type == "auto":
            if len(df.columns) == 1:
                chart_type = "histogram"
            elif len(df.columns) >= 2:
                chart_type = "bar" if len(df) < 20 else "line"

        # Generate chart
        if chart_type == "bar":
            fig = px.bar(df, x=df.columns[0], y=df.columns[1])
        elif chart_type == "line":
            fig = px.line(df, x=df.columns[0], y=df.columns[1])
        elif chart_type == "pie":
            fig = px.pie(df, names=df.columns[0], values=df.columns[1])
        else:  # histogram as default
            fig = px.histogram(df, x=df.columns[0])

        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

# Response Generator (Fixed prompt template)
# Updated get_response function
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    try:
        # Get clean SQL query
        sql_chain = get_sql_chain(db)
        query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history
        })
        
        # Debug output
        st.code(f"Executing SQL:\n{query}", language="sql")
        
        # Execute query
        result = db.run(query)
        
        # Generate analysis with proper prompt formatting
        response_template = """Analyze these SQL results:
        Question: {question}
        SQL: {query}
        Results: {response}
        
        Provide:
        1. Natural language summary
        2. Recommended visualization type (bar/line/pie/none)"""
        
        prompt = ChatPromptTemplate.from_template(response_template)
        
        # Create proper prompt input
        formatted_prompt = prompt.format(
            question=user_query,
            query=query,
            response=result
        )
        
        llm = ChatGroq(model="llama3-70b-8192", temperature=0)
        
        # Invoke LLM with proper input type
        response = llm.invoke(formatted_prompt)
        return response.content
        
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit App
def main():
    load_dotenv()
    st.set_page_config(page_title="MySQL Chat", page_icon="💬")
    st.title("💬 Chat with MySQL")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your MySQL assistant. Ask me anything about the database.")
        ]
    
    # Sidebar Connection
    with st.sidebar:
        st.subheader("Database Connection")
        host = st.text_input("Host", "localhost")
        port = st.text_input("Port", "3306")
        user = st.text_input("Username", "root")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database", "Chinook")
        
        if st.button("Connect"):
            with st.spinner("Connecting..."):
                try:
                    st.session_state.db = init_database(user, password, host, port, database)
                    st.success("Connected successfully!")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
    
    # Display Chat
    for message in st.session_state.chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.markdown(message.content)
    
    # Handle Query
    if user_query := st.chat_input("Ask about your data"):
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
        
        with st.chat_message("AI"):
            if 'db' not in st.session_state:
                st.error("Please connect to database first")
            else:
                try:
                    # Get response
                    response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                    
                    # Extract visualization recommendation
                    if "bar chart" in response.lower():
                        sql_chain = get_sql_chain(st.session_state.db)
                        query = sql_chain.invoke({
                            "question": user_query,
                            "chat_history": st.session_state.chat_history
                        })
                        data = st.session_state.db.run(query)
                        fig = create_visualization(data, "bar")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(response)
                    st.session_state.chat_history.append(AIMessage(content=response))
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))

if __name__ == "__main__":
    main()