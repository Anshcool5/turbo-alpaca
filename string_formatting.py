import ast
import re

text = """
To find the relevant keys from the dictionary for each metric in the list, we need to match the metric names with the keys in the dictionary. Here's the result:

Result = {
    'Total Sales': None, 
    'Gross Sales': 'Gross Sales', 
    'Net Sales': 'Total Net Sales', 
    'Total Orders': None, 
    'Discounts': 'Discounts', 
    'Returns': 'Returns', 
    'Shipping': None, 
    'customer_id': None, 
    'product_id': None, 
    'quantity': 'Net Quantity', 
    'date': None, 
    'Year': None, 
    'Month': None, 
    'cost_price': None, 
    'stock_level': None, 
    'expiry_date': None
}
"""

#code_blocks = re.findall(r"```(?:py(?:thon)?)\n(.*?)```", text, re.DOTALL)
#print(code_blocks)

pattern = r"Result\s*=\s*(\{.*?\})"
match = re.search(pattern, text, re.DOTALL)

if match:
    result_text = match.group(1)
    print("Extracted Dict:")
    print(dict(ast.literal_eval(result_text)))
else:
    raise Exception("Key Extraction failed!")

'''
for val in code_blocks:
    dic = val.split(" = ")[1]
    print(dic)
    try:
        temp = dict(ast.literal_eval(dic))
    except Exception:
        continue
    
print(f"Cur dic: {temp}")
'''
