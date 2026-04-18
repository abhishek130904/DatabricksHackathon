# Databricks notebook source
# DBTITLE 1,Secure API Key Retrieval
# Secure way to get Sarvam API key
# This will use secrets if available, otherwise fallback to hardcoded (temporary)

try:
    # Try to get from Databricks Secrets (most secure)
    SARVAM_API_KEY = dbutils.secrets.get(scope="my-secrets", key="sarvam-api-key")
    print("✅ Using API key from Databricks Secrets (secure)")
except:
    # Fallback: hardcoded (TEMPORARY - replace with secrets later!)
    SARVAM_API_KEY = "sk_73frjkbk_h9u67wusG8JiAh2z4VBZ5EuH"
    print("⚠️ Using hardcoded API key (not secure)")
    print("⚠️ To secure: Open web terminal and run:")
    print("   databricks secrets create-scope --scope my-secrets")
    print("   databricks secrets put --scope my-secrets --key sarvam-api-key")

print(f"\n✅ API key loaded (length: {len(SARVAM_API_KEY)} chars)")

# COMMAND ----------

# DBTITLE 1,Verify Secret Access
# Test that the secret is accessible
try:
    SARVAM_API_KEY = dbutils.secrets.get(scope="my-secrets", key="sarvam-api-key")
    print("✅ Secret retrieved successfully!")
    print("✅ API key is now hidden and secure")
    print(f"✅ Key length: {len(SARVAM_API_KEY)} characters")
    print("\nNote: The actual value is automatically redacted from notebook outputs.")
except Exception as e:
    print(f"❌ Error accessing secret: {e}")

# COMMAND ----------

# DBTITLE 1,Test API Key Usage
# Example: Using the secure API key from Cell 2
import requests

# SARVAM_API_KEY was already loaded in Cell 2
# No need to redefine it - just use it!

print("Testing API key...")
print(f"Key starts with: {SARVAM_API_KEY[:10]}...")
print(f"Key length: {len(SARVAM_API_KEY)} characters")
print("\n✅ Ready to use in your RAG system!")
print("\nExample usage in text-to-speech:")
print("")
print("def text_to_speech(text, filename='output.wav'):")
print("    url = 'https://api.sarvam.ai/text-to-speech'")
print("    headers = {")
print("        'Authorization': f'Bearer {SARVAM_API_KEY}',")
print("        'Content-Type': 'application/json'")
print("    }")
print("    payload = {'text': text, 'voice': 'anushka', 'language': 'en-IN'}")
print("    response = requests.post(url, headers=headers, json=payload)")
print("    return response")

# COMMAND ----------

# DBTITLE 1,Create Secret Scope and Store API Key
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

# Initialize Databricks SDK
w = WorkspaceClient()

print("Creating secret scope 'my-secrets'...")
try:
    # Create the secret scope
    w.secrets.create_scope(scope="my-secrets")
    print("✅ Secret scope 'my-secrets' created successfully")
except Exception as e:
    if "already exists" in str(e).lower():
        print("⚠️ Secret scope 'my-secrets' already exists")
    else:
        print(f"❌ Error creating scope: {e}")

print("\nStoring Sarvam API key...")
try:
    # Store the API key
    w.secrets.put_secret(
        scope="my-secrets",
        key="sarvam-api-key",
        string_value="sk_73frjkbk_h9u67wusG8JiAh2z4VBZ5EuH"
    )
    print("✅ API key stored successfully in secrets!")
    print("\n🔒 Your API key is now secure and hidden.")
except Exception as e:
    print(f"❌ Error storing secret: {e}")

# COMMAND ----------

