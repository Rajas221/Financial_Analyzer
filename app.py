import streamlit as st
import pandas as pd
import pdfplumber
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import plotly.express as px
import re
from google.api_core import exceptions

# --- Page Configuration ---
# Sets the title, layout, and initial state for the Streamlit page.
st.set_page_config(
    page_title="AI Financial Statement Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Environment Variables & Configure API ---
# Loads the API key from the .env file for secure access.
load_dotenv()
try:
    # Retrieve the API key from environment variables.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_GOES_HERE":
        st.error("⚠️ Gemini API Key not found. Please create a `.env` file and add your key.")
        st.stop()
    # Configure the Gemini generative AI model.
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
except Exception as e:
    st.error(f"Failed to configure the Gemini API: {e}")
    st.stop()

# --- Core Processing Functions ---

def extract_text_from_pdf(uploaded_file):
    """Extracts raw text from the uploaded PDF file."""
    text = ""
    try:
        # Open the PDF file using pdfplumber.
        with pdfplumber.open(uploaded_file) as pdf:
            # Iterate through each page and extract text.
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def get_structured_data_from_llm(pdf_text):
    """Sends text to Gemini LLM to get structured account and transaction data."""
    # This detailed prompt instructs the LLM on the exact JSON structure required.
    prompt = f"""
    You are an expert financial data extraction bot. Your task is to analyze the following bank statement text and convert it into a structured JSON format.

    The JSON output must have two top-level keys: "account_information" and "transactions".

    1. **account_information**: A single JSON object with these exact keys: "bank_name", "account_holder_name", "account_type", "account_number", "ifsc_code", "micr_code", "full_address". For "account_type", specify "Savings" or "Current". If a value isn't found, use "Not Available".

    2. **transactions**: A JSON array of objects. Each object must have keys: "date" (YYYY-MM-DD), "description", "withdrawal_amount" (numeric, 0 if none), "deposit_amount" (numeric, 0 if none), "closing_balance" (numeric).

    Analyze this text:
    ---
    {pdf_text}
    ---
    Provide only the JSON object.
    """
    try:
        # Display a spinner with the user-requested message while waiting for the API.
        spinner_message = "Document is getting analyzed. Please be patient as this may take a 3-5 minutes for large files."
        with st.spinner(spinner_message):
            # Generate content using the LLM. Timeout is removed as per user request.
            response = model.generate_content(prompt)
            # Clean the response to ensure it is valid JSON before parsing.
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_text)
    except Exception as e:
        # Handle errors during API communication.
        st.error(f"AI communication error: {e}")
        st.code(response.text if 'response' in locals() else "No response received.")
        return None

def analyze_and_flag_transactions(df):
    """Analyzes transaction data and flags entries with descriptive reasons."""
    if df.empty:
        return df, pd.DataFrame()

    df['flags'] = ''  # Initialize an empty string column for flags.
    flag_summary = []

    # Define the criteria for flagging transactions.
    criteria = {
        "DD Large Withdrawal": (df['description'].str.contains(r'\bDD\b', case=False, na=False)) & (df['withdrawal_amount'] > 10000),
        "RTGS Large Deposit": (df['description'].str.contains(r'\bRTGS\b', case=False, na=False)) & (df['deposit_amount'] > 50000),
        "Specific Entity Transaction": df['description'].str.contains(r'\b(Guddu|Prabhat|Arif|Coal India)\b', case=False, na=False)
    }

    # Iterate through criteria to apply flags and summarize findings.
    for flag_name, condition in criteria.items():
        df.loc[condition, 'flags'] += f"[{flag_name}] "
        count = condition.sum()
        if count > 0:
            flag_summary.append({"Alert Type": flag_name, "Count": count})
    
    # Create a new dataframe containing only the flagged transactions.
    flagged_transactions = df[df['flags'] != ''].copy()
    return flagged_transactions, pd.DataFrame(flag_summary)

def create_timeline_visualization(df):
    """Generates a timeline plot of withdrawals and deposits using Plotly Express."""
    if df.empty or 'date' not in df.columns:
        return None
    
    df_vis = df.copy()
    df_vis['date'] = pd.to_datetime(df_vis['date'])
    
    # Restructure the dataframe for easier plotting.
    df_melted = df_vis.melt(id_vars='date', value_vars=['deposit_amount', 'withdrawal_amount'],
                           var_name='transaction_type', value_name='amount')
    df_melted = df_melted[df_melted['amount'] > 0] # Exclude zero-amount entries.

    # Create the bar chart.
    fig = px.bar(df_melted, x='date', y='amount', color='transaction_type',
                 title='Withdrawals and Deposits Over Time', # User-requested static title.
                 labels={'date': 'Date', 'amount': 'Amount (INR)', 'transaction_type': 'Transaction Type'},
                 color_discrete_map={'deposit_amount': '#28a745', 'withdrawal_amount': '#dc3545'},
                 barmode='group')
    fig.update_layout(xaxis_title="Date", yaxis_title="Amount (INR)")
    return fig

# --- Streamlit App UI ---
# Main interface of the application starts here.
st.title("Financial Statement Analyzer") # User-requested title.
st.markdown("Upload any bank statement PDF to extract, analyze, and visualize its financial data.")

# Initialize session state to store data between reruns.
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Create the file uploader widget.
uploaded_file = st.file_uploader(
    "Choose a bank statement PDF",
    type="pdf",
    label_visibility="collapsed"
)

# Main logic block that runs when a file is uploaded.
if uploaded_file:
    with st.spinner("Processing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    if raw_text:
        llm_data = get_structured_data_from_llm(raw_text)

        if llm_data:
            st.success("Extraction Complete!")
            # Convert the JSON response from the LLM into pandas DataFrames.
            account_df = pd.DataFrame([llm_data.get("account_information", {})])
            transactions_df = pd.DataFrame(llm_data.get("transactions", []))
            
            # Ensure numeric columns have the correct data type.
            for col in ['withdrawal_amount', 'deposit_amount', 'closing_balance']:
                transactions_df[col] = pd.to_numeric(transactions_df[col], errors='coerce').fillna(0)

            # Perform analysis and visualization on the extracted data.
            flagged_df, summary_df = analyze_and_flag_transactions(transactions_df)
            timeline_fig = create_timeline_visualization(transactions_df)

            # Store all results in the session state.
            st.session_state.processed_data = {
                "account_df": account_df,
                "transactions_df": transactions_df,
                "flagged_df": flagged_df,
                "summary_df": summary_df,
                "timeline_fig": timeline_fig,
                "file_name": uploaded_file.name
            }

# This block displays the results if they are available in the session state.
if st.session_state.processed_data:
    data = st.session_state.processed_data
    base_filename = os.path.splitext(data['file_name'])[0]
    
    # Create tabs to organize the output.
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Account Info & Downloads", "📈 Transaction Analysis", "📊 Visualization", "🔍 All Transactions"])

    with tab1:
        st.subheader("Account Information")
        st.dataframe(data['account_df'], use_container_width=True)
        st.markdown("---")
        st.subheader("📥 Download Extracted Data")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Account Info (CSV)",
                data['account_df'].to_csv(index=False).encode('utf-8'),
                f"{base_filename}_account_info.csv", "text/csv", use_container_width=True
            )
        with col2:
             st.download_button(
                "Download All Transactions (CSV)",
                data['transactions_df'].to_csv(index=False).encode('utf-8'),
                f"{base_filename}_transactions.csv", "text/csv", use_container_width=True
            )

    with tab2:
        st.subheader("Transaction Analysis & Flagging")
        
        # Display the flagging criteria in an expandable section.
        with st.expander("View Flagging Criteria"):
            st.markdown("""
            Transactions are flagged if they meet one or more of the following conditions:
            - **DD Large Withdrawal**: The description contains "DD" and the withdrawal amount is over INR 10,000.
            - **RTGS Large Deposit**: The description contains "RTGS" and the deposit amount is over INR 50,000.
            - **Specific Entity Transaction**: The description contains "Guddu", "Prabhat", "Arif", or "Coal India" (case-insensitive).
            """)
        
        st.subheader("Flagging Summary")
        if not data['summary_df'].empty:
            st.dataframe(data['summary_df'], use_container_width=True)
            st.subheader("Flagged Transaction Details")
            st.dataframe(data['flagged_df'], use_container_width=True)
        else:
            st.info("No transactions matched the flagging criteria.")

    with tab3:
        st.subheader("Transaction Timeline")
        if data['timeline_fig']:
            st.plotly_chart(data['timeline_fig'], use_container_width=True)
        else:
            st.warning("Could not generate visualization. Ensure transaction data was extracted correctly.")
            
    with tab4:
        st.subheader("Complete Transaction Log")
        st.dataframe(data['transactions_df'], use_container_width=True)
