import pandas as pd
from openai import OpenAI
import json
import time
import os
import re
import io
import base64
import concurrent.futures

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

# Define the detailed prompt template for the GPT model.
# The prompt instructs the model to search for the following fields:
# Company, Asset, Asset Target, Asset Type, Modality, Disease, Global Highest Phase, Indication, Mechanism/Technology.
# The response should be in valid JSON without any additional commentary.
def build_prompt(record, keywords, search_type):
    if search_type == "company":
        prompt = f"""
                    You are an expert in biopharmaceutical research with the ability to search the web for up-to-date information.
                    For the company "{record['company']}", please research and provide the following details:
                    1. Company: The full name of the company.
                    2. Asset: The name of the asset or product they are developing that is related to {' '.join(keywords)}.
                    3. Asset Target: The specific biological target or pathway the asset addresses.
                    4. Asset Type: The specific type of asset (e.g., small molecule, biologic, device, etc.).
                    5. Modality: The specific therapeutic modality being used (e.g., antibody, gene therapy, cell therapy, etc.).
                    6. Disease: The disease area or condition being targeted.
                    7. Global Highest Phase: The highest phase of clinical development reached globally (e.g., Phase I, Phase II, etc.).
                    8. Indication: The specific medical indication for which the asset is being developed.
                    9. Mechanism/Technology: The specific mechanism of action or technology underlying the asset, in as much detail as possible.

                    Please return the results as a JSON object with keys:
                    "Company", "Asset", "Asset Target", "Asset Type", "Modality", "Disease", "Global Highest Phase", "Indication", "Mechanism/Technology".

                    Return only the JSON object without any additional commentary. 
                    It is essential that you be as detailed, thorough, and accurate as possible. 
                    Consult as many sources as needed and use any reliable source.
                    If any information is not available, set its value to null.
        """
    if search_type == "deal":
        prompt = f"""
                    You are an expert in biopharmaceutical deal analysis with the ability to search the web for up-to-date information.
                    For the deal involving "{record['acquirer']}" and "{record['acquired_company']}" related to "{' '.join(keywords)}", 
                    please research and provide the following details:
                    1. Acquirer: The full name of the company that is acquiring or investing.
                    2. Target Company: The name of the company or asset being acquired or partnered with.
                    3. Deal Type: The type of deal (e.g., acquisition, strategic investment, partnership, joint venture, etc.).
                    4. Deal Value: The total monetary value of the deal, specifying the currency (e.g., USD, EUR) if available.
                    5. Payment Structure: Details on the payment structure (e.g., cash, stock, combination, earn-out, contingent value rights, etc.).
                    6. Financial Advisors: The names of any financial advisors or advisory firms involved.
                    7. Announcement Date: The date when the deal was announced.
                    8. Deal Terms: Key terms of the deal, including any contingencies, milestones, or special conditions.
                    9. Strategic Rationale: A summary of the strategic reasons behind the deal.
                    10. Additional Details: Any other pertinent financial or operational information related to the deal.

                    Please return the results as a JSON object with keys:
                    "Acquirer", "Target Company", "Deal Type", "Deal Value", "Payment Structure", "Financial Advisors", "Announcement Date", "Deal Terms", "Strategic Rationale", "Additional Details".

                    You also have the following as a resource, which you may consider factual information about the deal:
                    {record['combined_text']}

                    Return only the JSON object without any additional commentary.
                    It is essential that you be as detailed, thorough, and accurate as possible.
                    Consult as many online sources as needed and use any reliable source.
                    If any information is not available, set its value to null.
        """
    else:
        raise Exception(f"Invalid search type: {search_type}")

    return prompt.strip()

def extract_json_and_source(text):
    # Remove markdown code fences if present.
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    
    # Find the first '{' and the last '}' to capture the JSON block.
    json_start = text.find("{")
    json_end = text.rfind("}")
    if json_start == -1 or json_end == -1:
        raise ValueError("No JSON object found in the text.")
    
    # Extract the JSON portion.
    json_str = text[json_start:json_end+1]
    # The remaining text after the JSON block is considered the source.
    remaining = text[json_end+1:].strip()
    
    # Parse the JSON.
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError("Error parsing JSON: " + str(e))
    
    # If there's remaining text, add it as a new key.
    # (scrapping this for now)
    #if remaining:
    #    data["Source"] = remaining
    
    return data

# Function to call the GPT model and parse the returned JSON.
def gpt_prompt(record, keywords, max_retries=3):
    company_name = record["company"]
    prompt = build_prompt(record, keywords)
    
    # Set up the conversation for ChatCompletion (if using a chat-based model)
    messages = [
        {"role": "system", "content": "You are a research assistant that extracts specific information in JSON format."},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-search-preview",
                web_search_options={
                    "search_context_size": "high",
                },
                messages=messages
            )
            # Extract the assistant's message
            content = response.choices[0].message.content.strip()
            data = extract_json_and_source(content)
            print(data)
            return data
        except Exception as e:
            print(f"Error processing {company_name} on attempt {attempt+1}: {e}")
            time.sleep(2)  # wait before retrying
    # If all retries fail, return a dictionary with null values
    return {
        "Company": company_name,
        "Asset": None,
        "Asset Target": None,
        "Asset Type": None,
        "Modality": None,
        "Disease": None,
        "Global Highest Phase": None,
        "Indication": None,
        "Mechanism/Technology": None
    }

def enrich(records, keywords, search_types):
    """
    Enriches a list of companies concurrently for each search type and returns an Excel file
    (as a Base64-encoded string) with one sheet per search type.
    """
    search_type_dataframes = {}
    max_workers = 5

    # Process each search type separately.
    for search_type in search_types:
        enriched_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each company for the current search type.
            future_to_company = {
                executor.submit(gpt_prompt, record, keywords, search_type): record 
                for record in records
            }
            for future in concurrent.futures.as_completed(future_to_company):
                company = future_to_company[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"{company} generated an exception: {exc}")
                    result = {
                        "Company": company,
                        "Asset": None,
                        "Asset Target": None,
                        "Asset Type": None,
                        "Modality": None,
                        "Disease": None,
                        "Global Highest Phase": None,
                        "Indication": None,
                        "Mechanism/Technology": None
                    }
                enriched_data.append(result)
        # Convert the enriched data for this search type into a DataFrame.
        enriched_df = pd.DataFrame(enriched_data)
        search_type_dataframes[search_type] = enriched_df

    # Write each search type's DataFrame to its own sheet in the Excel workbook.
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for st, df in search_type_dataframes.items():
            # Ensure sheet names are within Excel's 31-character limit.
            sheet_name = st if len(st) <= 31 else st[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    excel_base64 = base64.b64encode(output.read()).decode('utf-8')
    return excel_base64