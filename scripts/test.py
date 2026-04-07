import yaml

with open(r"C:/Users/louay/Downloads/olives_2/data.yaml", 'r') as f:
    data = yaml.safe_load(f)

print(data)