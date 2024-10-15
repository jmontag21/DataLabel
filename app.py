import os
import streamlit as st
from openai import OpenAI
from pdf2image import convert_from_path
import pandas as pd
import json
import base64
import time
import re
import tempfile

# Initialize Streamlit session state for data caching
if 'data' not in st.session_state:
    st.session_state.data = None

if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []

# Set up the Streamlit app
st.title("Data Label Invoice PDF Processor")
st.write("""
Upload your Data Label PDF invoices, and this app will extract relevant information and provide a downloadable CSV file.
""")

# Access the API key from Streamlit Secrets
try:
    API_KEY = st.secrets["openai"]["api_key"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets. Please configure your secrets.toml file.")
    st.stop()

# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)

# File uploader allows multiple files
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Get the list of uploaded file names to track changes
    current_uploaded_file_names = sorted([file.name for file in uploaded_files])

    # Check if the uploaded files have changed
    if st.session_state.data is None or st.session_state.uploaded_file_names != current_uploaded_file_names:
        # Update the list of uploaded file names in session state
        st.session_state.uploaded_file_names = current_uploaded_file_names
        # Reset the cached data
        st.session_state.data = None

    if st.session_state.data is None:
        with st.spinner("Setting up..."):
            # Create a temporary directory to store uploaded PDFs and temporary images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded PDFs to the temporary directory
                pdf_files = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_files.append(uploaded_file.name)
                
                # Debugging: Print the number of PDF files found
                st.write(f"Found {len(pdf_files)} PDF file(s).")

                # If no PDF files are found, display an error
                if not pdf_files:
                    st.error("No PDF files found. Please upload at least one PDF.")
                
                # Field name mapping
                field_name_mapping = {
                    'INVOICE DATE': 'INVOICE_DATE',
                    'INVOICE NUMBER': 'INVOICE_NUMBER',
                    'CUST. PO#': 'CUSTOMER_PO',
                    'SUB-TOTAL:': 'SUB_TOTAL',
                    'FREIGHT:': 'FREIGHT',
                    'TOTAL:': 'TOTAL',
                    'TRACKING/PRO NUMBER': 'TRACKING_NUMBER',
                }
                
                def normalize_field_names(data_dict):
                    normalized_dict = {}
                    for key, value in data_dict.items():
                        normalized_key = field_name_mapping.get(key.strip().upper(), key.strip().upper())
                        normalized_dict[normalized_key] = value
                    return normalized_dict
                
                def extract_tracking_number(text):
                    # Look for patterns starting with "1Z" followed by alphanumeric characters
                    match = re.search(r'1Z[A-Z0-9]{16}', text)
                    return match.group(0) if match else "No Tracking Number Found"
                
                def process_pdf(pdf_file, max_retries=3, retry_delay=5):
                    pdf_path = os.path.join(temp_dir, pdf_file)
                    
                    for attempt in range(max_retries):
                        try:
                            # Convert the first page of the PDF to an image
                            images = convert_from_path(pdf_path, first_page=1, last_page=1, fmt='png', output_folder=temp_dir, single_file=True)
                            image = images[0]
                    
                            # Save the image temporarily
                            temp_image_name = f'temp_{pdf_file}_{attempt}.png'
                            image_path = os.path.join(temp_dir, temp_image_name)
                            image.save(image_path)
                    
                            # Encode the image in base64
                            with open(image_path, "rb") as image_file:
                                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                            # Prepare the prompt and messages
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "Please extract the following fields from the invoice image and provide the data in JSON format, using the exact field names provided, and without any code blocks or additional formatting:\n"
                                                "- INVOICE_DATE\n"
                                                "- INVOICE_NUMBER\n"
                                                "- CUSTOMER_PO (look for 'Cust. PO#')\n"
                                                "- SUB_TOTAL (look for 'Sub-total:')\n"
                                                "- FREIGHT\n"
                                                "- TOTAL\n"
                                                "- TRACKING_NUMBER (look for a number starting with '1Z' followed by alphanumeric characters)\n"
                                                "\nPlease include the full text content of the invoice in your response as well."
                                            )
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{base64_image}"
                                            }
                                        }
                                    ]
                                }
                            ]
                    
                            # Call the OpenAI API with image input
                            response = client.chat.completions.create(
                                model='gpt-4o',
                                messages=messages,
                                max_tokens=1000
                            )
                    
                            # Get the response text
                            response_text = response.choices[0].message.content
                            
                            # Parse the JSON response
                            try:
                                # Extract the JSON part from the response
                                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(0)
                                    extracted_data = json.loads(json_str)
                                else:
                                    raise ValueError("No JSON found in the response")
                                
                                # Normalize field names
                                extracted_data = normalize_field_names(extracted_data)
                                
                                # Extract tracking number from the full text content
                                extracted_data['TRACKING_NUMBER'] = extract_tracking_number(response_text)
                                
                                # Add the PDF file name to the extracted data
                                extracted_data['pdf_file'] = pdf_file
                                
                                # Remove the temporary image file
                                os.remove(image_path)
                                
                                return extracted_data  # Success, return the data
                            
                            except json.JSONDecodeError:
                                st.warning(f"Failed to parse JSON for file {pdf_file} on attempt {attempt + 1}")
                                st.info(f"Response text: {response_text}")
                                raise ValueError("JSON parsing failed")
                    
                        except Exception as e:
                            st.warning(f"An error occurred while processing {pdf_file} on attempt {attempt + 1}: {e}")
                            # Remove the temporary image file if it exists
                            if 'image_path' in locals() and os.path.exists(image_path):
                                os.remove(image_path)
                            if attempt < max_retries - 1:
                                st.info(f"Retrying {pdf_file} in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                            else:
                                st.error(f"All retries failed for {pdf_file}")
                                return None  # Return None after all retries have failed
                
                # Initialize a list of placeholders for status messages
                status_placeholders = [st.empty() for _ in pdf_files]
                
                # Initialize a progress bar
                progress_bar = st.progress(0)
                total_files = len(pdf_files)
                
                # Initialize an empty list to store the extracted data
                data = []
                
                # Process each PDF sequentially
                for idx, pdf_file in enumerate(pdf_files):
                    status_placeholders[0].markdown(f"**{pdf_file}**: Processing...")
                    result = process_pdf(pdf_file)
                    
                    if result is not None:
                        data.append(result)
                    else:
                        status_placeholders[0].error("Failed")
                    
                    # Update progress bar
                    progress_bar.progress((idx + 1) / total_files)
        
                # Final status update
                st.write("All files processed.")
                
                if data:
                    # Create a DataFrame from the extracted data
                    df = pd.DataFrame(data)
                    
                    # Reorder the columns if needed
                    columns_order = ['INVOICE_DATE', 'INVOICE_NUMBER', 'CUSTOMER_PO', 'SUB_TOTAL', 'FREIGHT', 'TOTAL', 'TRACKING_NUMBER', 'pdf_file']
                    df = df.reindex(columns=columns_order)
                    
                    # Save the DataFrame to a CSV file in memory
                    csv_buffer = df.to_csv(index=False).encode('utf-8')
                    
                    # Store the processed data in session state
                    st.session_state.data = data
                else:
                    st.error("No data was extracted from the uploaded PDFs.")
    
    else:
        st.info("Using cached data.")
    
    if st.session_state.data:
        with st.spinner("Preparing CSV..."):
            # Create a DataFrame from the session state data
            df = pd.DataFrame(st.session_state.data)
            
            # Reorder the columns if needed
            columns_order = ['INVOICE_DATE', 'INVOICE_NUMBER', 'CUSTOMER_PO', 'SUB_TOTAL', 'FREIGHT', 'TOTAL', 'TRACKING_NUMBER', 'pdf_file']
            df = df.reindex(columns=columns_order)
            
            # Save the DataFrame to a CSV file in memory
            csv_buffer = df.to_csv(index=False).encode('utf-8')
        
        # Provide a download button for the CSV
        st.success("Processing complete!")
        st.download_button(
            label="Download CSV",
            data=csv_buffer,
            file_name='data_label_invoices.csv',
            mime='text/csv',
        )
        
        # Optionally display the DataFrame
        st.dataframe(df)
    else:
        st.error("No data was extracted from the uploaded PDFs.")