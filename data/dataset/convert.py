import pandas as pd
import json

with open(r"C:\Users\avdho\.minimax-agent\projects\7\golden68_framework\data\dataset\golden68.json") as f:
    data = json.load(f)

df = pd.json_normalize(data)
df.to_csv("golden68.csv", index=False)