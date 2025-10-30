import requests
import re
import time
import pandas as pd
import matplotlib.pyplot as plt
import tabulate


app_base_url = "http://sentiment-analysis-env.eba-seu7awzm.us-east-1.elasticbeanstalk.com/predict-form"

test_cases = [
    ("Economy shows steady growth according to latest data.", "REAL"),
    ("Scientists confirm the earth is flat in new study.", "FAKE"),
    ("Local sports team wins championship after dramatic final.", "REAL"),
    ("Celebrity endorses miracle weight loss pill.", "FAKE")
]

results = []

# 100 API calls per test case
for case_text, expected in test_cases:
    for i in range(100):  
        start_time = time.time()
        try:
            # Check for the prediction and the expected value
            response = requests.post(app_base_url, data={"message": case_text})
            match = re.search(r'<span class="prediction-value">(.*?)</span>', response.text)
            prediction = match.group(1) if match else "No prediction"
            assert prediction == expected, f"Expected '{expected}', got '{prediction}'"
            
        except Exception as ex:
            print(f"Error occurred: {ex}")
        
        # calculate the latency and create an entry in results
        latency = time.time() - start_time
        results.append({"test_case": case_text, "predicted":prediction , "expected": expected, "latency": latency})

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("api_latency_results.csv", index=False)
print("CSV saved: api_latency_results.csv")

# boxplot
short_labels = {text: f"TC{i+1}" for i, (text, _) in enumerate(test_cases)}
df["test_case_short"] = df["test_case"].map(short_labels)
plt.figure(figsize=(10,6))
df.boxplot(column="latency", by="test_case_short", grid=True)
plt.title("API Latency per Test Case")
plt.suptitle("") 
plt.ylabel("Latency (seconds)")
plt.xlabel("Test Case")
plt.xticks(rotation=45)
plt.show()

# Calculate average latency per test case
avg_latency = df.groupby("test_case")["latency"].mean().round(4)
table_data = [[test_case, latency] for test_case, latency in avg_latency.items()]
print(tabulate.tabulate(table_data, headers=['Test Case', 'Average Latency (s)'], tablefmt="fancy_grid"))
