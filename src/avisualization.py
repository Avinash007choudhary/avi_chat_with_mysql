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
import matplotlib.pyplot as plt
from decimal import Decimal
import ast



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
    
    Write only the SQL query and show visualization if asked. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question:show me count of customers who did transaction to Mexico in the year 2024?
    SQL Query: select count(our_customer_id) from comp_financial_tran_repos_dly where counterparty_customer_country='Mexico' and year(transaction_date)='2024';
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
def parse_sql_result(data_str):
    """Parse SQL result string like "[('Houston', Decimal('25188006.27')), ...]"""
    pattern = r"\('([^']+)', Decimal\('([\d.]+)'\)\)"
    matches = re.findall(pattern, data_str)
    return [(country, float(amount)) for country, amount in matches]

def parse_sql_data(data):
    """Parse various SQL result formats into DataFrame"""
    if isinstance(data, pd.DataFrame):
        return data
    
    if isinstance(data, str):
        try:
            # Case 1: [('name', Decimal('123.45')] format
            if data.startswith("[(") and "Decimal" in data:
                pattern = r"\('([^']+)', Decimal\('([\d.]+)'\)\)"
                matches = re.findall(pattern, data)
                return pd.DataFrame(matches, columns=['category', 'value'])
            
            # Case 2: JSON format
            if data.startswith("{") or data.startswith("["):
                return pd.DataFrame(json.loads(data))
            
            # Case 3: Raw values
            return pd.DataFrame([float(x) for x in data.split('\n') if x.strip()])
        
        except:
            return pd.DataFrame()

    # Handle list/dict inputs
    return pd.DataFrame(data)


def recommend_chart_type(df):
    """Auto-select chart type based on data characteristics"""
    if len(df.columns) == 1:
        return "histogram"
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(df) > 100:
        return "scatter" if len(numeric_cols) >= 2 else "line"
    
    if len(df.columns) == 2:
        if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            return "line"
        return "bar"
    
    if len(numeric_cols) >= 2:
        return "scatter"
    
    return "bar"

def generate_chart(df, chart_type, title):
    """Generate different chart types dynamically"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number']).columns
    
    # Default column mappings
    x_col = cat_cols[0] if len(cat_cols) > 0 else df.columns[0]
    y_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[-1]
    
    if chart_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, title=title)
    elif chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, title=title)
    elif chart_type == "pie":
        fig = px.pie(df, names=x_col, values=y_col, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, title=title)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x_col, title=title)
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title)  # Default fallback
    
    # Auto-formatting
    if y_col in numeric_cols:
        fig.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.2f')
    fig.update_layout(hovermode='x unified')
    return fig


def create_dynamic_visualization(data, chart_type="auto", title="Data Visualization"):
    try:
        # Parse incoming data
        df = parse_sql_data(data)
        if df.empty:
            return None

        # Auto-detect best chart type if not specified
        if chart_type == "auto":
            chart_type = recommend_chart_type(df)

        # Generate visualization
        fig = generate_chart(df, chart_type, title)
        return fig

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
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
    
    #template = """
    #You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    #Based on the table schema, question, sql query, and sql response, write a natural language response.
    #When the user asks for data that would be better visualized (like comparisons, distributions, or trends), 
    #include a visualization instruction like this: [VISUALIZATION:chart_type] where chart_type can be bar, line, or pie.
    
    #<SCHEMA>{schema}</SCHEMA>
    #Conversation History: {chat_history}
    #SQL Query: <SQL>{query}</SQL>
    #User question: {question}
    #SQL Response: {response}"""
    
    template = """
    When responding to data queries, format your answer as:

        <ANALYSIS>
        [Your interpretation of the data]
        </ANALYSIS>

        <VISUALIZATION-TYPE>
        [Recommended chart type: bar|line|pie|scatter|histogram|auto]
        </VISUALIZATION-TYPE>

        <DATA-FORMAT>
        [x_column: column_name]
        [y_column: column_name (if applicable)]
        [title: Suggested chart title]
        </DATA-FORMAT>

        <SQL-DATA>
        [Raw query results]
        </SQL-DATA> 
        
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """


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

def handle_visualization(response):
    """Process LLM response and generate visualization"""
    # Extract components from LLM response
    vis_type = extract_between_tags(response, "VISUALIZATION-TYPE") or "auto"
    data = extract_between_tags(response, "SQL-DATA")
    title = extract_between_tags(response, "title") or "Data Visualization"
    
    # Generate and display chart
    fig = create_dynamic_visualization(data, chart_type=vis_type, title=title)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not generate visualization from the data")

def extract_between_tags(text, tag_name):
    """Helper to extract content between XML-like tags"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


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
                fig = create_dynamic_visualization(message.visualization_data)
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
        if 'db' not in st.session_state or st.session_state.db is None:
            st.error("Please connect to the database first")
            st.session_state.chat_history.append(AIMessage(content="Please connect to the database first."))
        else:
            try:
                # Get the initial response
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                
                # Check if visualization is requested
                if '[VISUALIZATION:' in response:
                    try:
                        # Extract visualization instruction safely
                        vis_match = re.search(r'\[VISUALIZATION:(\w+)\]', response)
                        if vis_match:
                            chart_type = vis_match.group(1)  # Define chart_type here
                            response_text = response.replace(vis_match.group(0), "")
                            
                            # Get the SQL query
                            sql_chain = get_sql_chain(st.session_state.db)
                            query = sql_chain.invoke({
                                "question": user_query,
                                "chat_history": st.session_state.chat_history
                            })
                            
                            # Get the data
                        try:
                            data = st.session_state.db.run(query)
    
                            # Convert Decimal to float
                            if isinstance(data, str):
                                try:
                                    data = json.loads(data, parse_float=lambda x: float(x))
                                except:
                                    # Handle case where data isn't JSON
                                    pass

                            fig = create_dynamic_visualization(data, chart_type, title="Data Visualization")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Update chat history
                                st.session_state.chat_history.append(AIMessage(content=response_text))
                            else:
                                st.markdown(response)
                                st.session_state.chat_history.append(AIMessage(content=response))
                        except Exception as e:
                            st.error(f"Error running query or creating visualization: {str(e)}")
                            st.markdown(response)
                            st.session_state.chat_history.append(AIMessage(content=f"{response}\n\nError: {str(e)}"))
                    
                    except Exception as e:
                        error_msg = f"🚨 Failed to generate visualization: {str(e)}"
                        st.error(error_msg)
                        st.markdown(response)
                        st.session_state.chat_history.append(AIMessage(content=f"{response}\n\n{error_msg}"))
                
                else:
                    # No visualization requested
                    st.markdown(response)
                    st.session_state.chat_history.append(AIMessage(content=response))
            
            except Exception as e:
                error_msg = f"🚨 Error processing your request: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(AIMessage(content=error_msg))