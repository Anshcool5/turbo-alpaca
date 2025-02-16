import requests

url = ''
response = requests.get(url)
html_content = response.text

print(html_content)