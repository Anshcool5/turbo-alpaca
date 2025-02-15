import csv
import json

# Specify the CSV file path
csv_file_path = "sample_data/business.retailsales.csv"
json_file_path = "data_for_chroma/business.retailsales.json"

# Read the CSV and convert to JSON
data = []
with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        data.append(row)

# Write the JSON to a file
with open(json_file_path, mode="w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=4)

print("CSV converted to JSON successfully.")