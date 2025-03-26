import pandas as pd
import openai
from openai import OpenAI
import json
import time
import config
import os
import re

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

# Define the detailed prompt template for the GPT model.
# The prompt instructs the model to search for the following fields:
# Company, Asset, Asset Target, Asset Type, Modality, Disease, Global Highest Phase, Indication, Mechanism/Technology.
# The response should be in valid JSON without any additional commentary.
def build_prompt(company_name):
    prompt = f"""
                You are an expert in biopharmaceutical research with the ability to search the web for up-to-date information.
                For the company "{company_name}", please research and provide the following details:
                1. Company: The full name of the company.
                2. Asset: The name of the asset or product they are developing.
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
                Consult as many sources as needed. Use any reliable source.
                If any information is not available, set its value to null.
"""
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
def gpt_prompt(company_name, max_retries=3):
    prompt = build_prompt(company_name)
    
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
                    "search_context_size": "low",
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

def enrich(companies):
    """
    Enriches a list of companies and saves the result as an Excel file.
    Returns the file path to the generated Excel file.
    """
    enriched_data = []
    for company_name in companies:
        print(f"Processing {company_name}...")
        enriched_info = gpt_prompt(company_name)
        enriched_data.append(enriched_info)
        time.sleep(1)  # Respect rate limits if needed

    enriched_df = pd.DataFrame(enriched_data)
    output_file = "data.xlsx"
    try:
        enriched_df.to_excel(output_file, index=False)
        print(f"Enriched data has been saved to {output_file}")
    except Exception as e:
        print(f"Error writing to Excel file: {e}")
    return output_file