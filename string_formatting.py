import ast
import re

text = """
**Evaluation of Business Idea: Bubble Tea Shop with Taiwanese Fried Chicken**

1. **Risk**: 6/10  
   The market for bubble tea is saturated, posing a moderate risk. However, the addition of Taiwanese fried chicken offers a unique twist that can help mitigate this risk by differentiating the shop from competitors.

2. **Competitiveness**: 7/10  
   While the bubble tea market is competitive, the combination with fried chicken provides a unique selling proposition. Success will depend on quality and effective marketing to stand out.

3. **Setup Cost**: 7/10  
   High initial investment is required due to the need for kitchen equipment, a good location, and staff training. However, some costs can be shared between the two offerings.

4. **Expected ROI**: 7/10  
   Both products have good profit margins, but ROI depends on location, customer attraction, and sales volume. Effective execution is crucial for profitability.

5. **Scalability**: 8/10  
   The concept is scalable, with potential for expansion or franchising. Maintaining consistency across locations will be key to successful scaling.

This assessment highlights a moderate to high potential business idea with unique elements that can attract a loyal customer base, provided execution and location are well-managed.
"""

#code_blocks = re.findall(r"```(?:py(?:thon)?)\n(.*?)```", text, re.DOTALL)
#print(code_blocks)

pattern = r"\d+\.\s*\*\*(.*?)\*\*"

matches = re.findall(pattern, text)
print(matches)

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
