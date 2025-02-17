from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


load_dotenv()

# Get the API Key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b-specdec", groq_api_key=groq_api_key)

def determine_and_call_analytics(query: str, key_list: list):

    metrics_input = f"""Based on the user query: '{query}' and the list of metrics {key_list}, determine if the requested
    plots can be generated or not. If they ALL can be generated, return YES. If not all but some plots can be generated, 
    return SOME along with the name of the plots that can be generated. Else return NO along with needed metrics to process the user request."""

    metrics_template = """
    Human: {text}
    Assistant: return the word YES or SOME along with the name of the plots that can be generated or NO along with the needed metrics. 
    """

    metrics_prompt = PromptTemplate(
        template=metrics_template,
        input_variables=["text"]
    )

    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa1 = LLMChain(llm=llm, prompt=metrics_prompt)
    result1 = qa1.generate([{"text": metrics_input}])
    output = result1.generations[0][0].text
    if output.contains("YES"):
        output = "Sure thing, I'll be generating your plots based on your shared files!"
    elif output.contains("SOME"):
        output = "Based on the given data, I can only generate some, and I'll do so!"
    elif output.contains("NO"):
        output = "I'm sorry your shared data is either missing key metrics or the  plot is out of my current scope."

    return output