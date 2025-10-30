import requests
import re

app_base_url = "http://sentiment-analysis-env.eba-seu7awzm.us-east-1.elasticbeanstalk.com/predict-form"

test_cases = [
    ("Economy shows steady growth according to latest data.", "REAL" ), # real
    ("An asteroid will hit Earth tomorrow.", "FAKE"), # fake
    ("Scientist have recently discovered 2 Moons.", "FAKE"), # fake
    ("Global warming is causing temperatures to rise.", "REAL") # real
]

print("Starting functional testing...\n")

successful = 0
error = 0
for case in test_cases:
    try:
        response = requests.post(app_base_url, data={"message": case})
        match = re.search(r'<span class="prediction-value">(.*?)</span>', response.text)
        if match:
            prediction = match.group(1)
        assert prediction == case[1], f"Test failed: expected '{case[1]}' but got '{prediction}' for input '{case[0]}'"
        successful += 1

    except Exception as ex:
        print(f"Error occured: {ex}\n")
        error += 1


print(f"Functional testing completed - {successful}/{len(test_cases)} successful. {error} failed.")
