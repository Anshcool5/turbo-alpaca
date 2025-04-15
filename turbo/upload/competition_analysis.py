import time
from playwright.sync_api import sync_playwright
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
from bs4 import BeautifulSoup
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get the API Key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

def get_search_query_from_llm(query):
    prompt = """You're a robust and well trained region of interest indetifier based on plain english query.
    English query Schema:
    "I want to open a bakery in Southgate Edmonton, can you please perform a competition analysis"
    Sample Output:
    "bakery, Southgate, edmonton, AB"
    You will reply with the same format as the sample output, i.e., "business, region, city, province"
    ---
    Input: {query}
    """

    search_query_prompt = PromptTemplate(
        template=prompt,
        input_variables=["query"]
    )
    qa1 = LLMChain(llm=llm, prompt=search_query_prompt)
    result1 = qa1.generate([{"query": query}])
    response = result1.generations[0][0].text.strip()
    #print(f"RESPONSE: {response}")
    return response

def run_playwright(search_query):
    base_url = 'https://maps.google.com'

    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(java_script_enabled=True)
    page = context.new_page()
    page.goto(base_url, wait_until='load')

    # find the search box
    input_box = page.locator('//input[@name="q"]')
    input_box.fill(search_query)
    input_box.press('Enter')

    xpath_search_result_element = '//div[@role="feed"]'
    page.wait_for_selector(xpath_search_result_element)
    results_container = page.query_selector(xpath_search_result_element)
    results_container.scroll_into_view_if_needed()

    # Resolve path relative to this script's file
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    #html_path = os.path.join(script_dir, "html_for_analysis", "maps.html")
    
    keep_scrolling = True
    while keep_scrolling:
        results_container.press('Space')
        time.sleep(2.5)
        if results_container.query_selector('//span[text()="You\'ve reached the end of the list."]'):
            results_container.press('Space')
            keep_scrolling = False

        #os.makedirs(os.path.dirname(html_path), exist_ok=True)  # Ensure folder exists

        #with open(html_path, 'w', encoding='utf-8') as f:
        #    f.write(results_container.inner_html())
    html_content = results_container.inner_html()

    context.close()
    browser.close()
    playwright.stop()

    return html_content

def extract_text_elements(soup, tag, attributes):
    elements = soup.find_all(tag, attributes)
    return [element.text for element in elements if element.text]

def generate_json_prompt():
    return """Convert the following list into a JSON object with each records based on this \
    JSON record schema:
    {{
        "name": "Bill Kim Ramen Bar",
        "rating": "3.2",
        "reviews": "45",
        "price": "$$",
        "category": "Ramen",
        "location": "916 W Fulton Market",
        "hours": "Open . Closes 9PM",
        "services": [
            "DIne-in",
            "Takeout",
            "Delivery"
        ],
        "actions": [
            "Order Online"
        ]
    }}
    You will reply only with the jSON itself, and no other descriptive or explanatory text
    ---
    Input: {input_list}
    """

def get_llm_response(llm, prompt, input_list):
    json_prompt = PromptTemplate(
        template=prompt,
        input_variables=["input_list"]
    )
    qa1 = LLMChain(llm=llm, prompt=json_prompt)
    result1 = qa1.generate([{"input_list": input_list}])
    response = result1.generations[0][0].text.strip()
    #print(f"RESPONSE: {response}")
    return response

def html_2_csv(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    results = extract_text_elements(soup, 'div', {'jsaction': True})

    recordset = []
    batch_size = 35
    for i in range(0, 100, batch_size):
        input_list = results[i:i+batch_size]
        prompt = generate_json_prompt()

        print("Running..")
        response_content = get_llm_response(llm, prompt, input_list)
        data = json.loads(response_content)

        print(len(data))
        recordset.extend(data)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(parent_dir, "media/csv_for_upload", "competition_analysis.csv")
    df = pd.DataFrame(recordset)
    df.to_csv(csv_path, encoding='utf-8-sig', index=False)

if __name__ == "__main__":
    user_query = "I want to open an Indian Restaurant in Jasper Ave, Edmonton, can you perform a competition analysis for me?"
    response = get_search_query_from_llm(user_query)
    print(f"RESPONSE: {response}")
    parsed_html = run_playwright(response)
    html_2_csv(parsed_html)