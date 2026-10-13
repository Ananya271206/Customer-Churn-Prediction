import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "telco_customer_churn.csv")

def download_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        print(f"Dataset already exists at {OUTPUT_FILE}")
        return OUTPUT_FILE

    print(f"Downloading Telco Customer Churn dataset from {DATA_URL}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(DATA_URL, headers=headers)
    with urllib.request.urlopen(req) as response, open(OUTPUT_FILE, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    
    print(f"Successfully downloaded to {OUTPUT_FILE} ({len(data)} bytes)")
    return OUTPUT_FILE

if __name__ == "__main__":
    download_data()
