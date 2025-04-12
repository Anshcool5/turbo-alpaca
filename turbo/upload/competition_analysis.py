import time
from playwright.sync_api import sync_playwright
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(script_dir, "html_for_analysis", "maps.html")

    keep_scrolling = True
    while keep_scrolling:
        results_container.press('Space')
        time.sleep(2.5)
        if results_container.query_selector('//span[text()="You\'ve reached the end of the list."]'):
            results_container.press('Space')
            keep_scrolling = False

        os.makedirs(os.path.dirname(html_path), exist_ok=True)  # Ensure folder exists

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(results_container.inner_html())

    context.close()
    browser.close()
    playwright.stop()

if __name__ == "__main__":
    user_query = "I want to open an Indian Restaurant in Jasper Ave, Edmonton, can you perform a competition analysis for me?"
    response = get_search_query_from_llm(user_query)
    print(f"RESPONSE: {response}")
    run_playwright(response)