import pandas as pd
import json

df = pd.read_pickle("processed_reviews.pkl")

clean = df[["asin", "reviewTime", "overall", "fnet_embedding"]].dropna()

clean["fnet_embedding"] = clean["fnet_embedding"].apply(lambda x: json.dumps(x.tolist()))

clean.to_csv("clean_reviews.csv", index=False)

print("✅ clean_reviews.csv created with", len(clean), "rows (embeddings in JSON format)")
