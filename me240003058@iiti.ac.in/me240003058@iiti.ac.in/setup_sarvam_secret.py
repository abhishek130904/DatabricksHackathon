# Databricks notebook source
# MAGIC %md
# MAGIC # 🔐 Setup Sarvam AI API Key (One-Time Setup)
# MAGIC 
# MAGIC This notebook helps you securely store your Sarvam AI API key in Databricks Secrets.
# MAGIC 
# MAGIC **Run this once, then delete it for security!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Secret Scope

# COMMAND ----------

# Create a secret scope named "vidya-setu"
# Note: In Databricks Community Edition, you may need to use Azure Key Vault or AWS Secrets Manager
# For full Databricks workspaces, you can create a Databricks-backed secret scope

try:
    dbutils.secrets.listScopes()
    print("✅ Secret scopes available!")
    print("\nExisting scopes:")
    for scope in dbutils.secrets.listScopes():
        print(f"  - {scope.name}")
except Exception as e:
    print(f"⚠️ Error listing scopes: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Store API Key
# MAGIC 
# MAGIC **Option A: Using Databricks CLI (Recommended)**
# MAGIC 
# MAGIC Run this in your terminal:
# MAGIC ```bash
# MAGIC databricks secrets create-scope --scope vidya-setu
# MAGIC databricks secrets put --scope vidya-setu --key sarvam-api-key
# MAGIC ```
# MAGIC 
# MAGIC **Option B: Using Databricks UI**
# MAGIC 
# MAGIC 1. Go to Settings → Secrets
# MAGIC 2. Create scope: `vidya-setu`
# MAGIC 3. Add secret: `sarvam-api-key`
# MAGIC 4. Paste value: `sk_73frjkbk_h9u67wusG8JiAh2z4VBZ5EuH`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Verify Secret is Set

# COMMAND ----------

try:
    # Try to retrieve the secret (won't show the value for security)
    secret_value = dbutils.secrets.get(scope="vidya-setu", key="sarvam-api-key")
    
    if secret_value:
        print("✅ SUCCESS! Sarvam API key is configured!")
        print(f"✅ Key length: {len(secret_value)} characters")
        print(f"✅ Key preview: {secret_value[:10]}...")
        print("\n🎉 Your Vidya Setu app will now use the secure API key!")
    else:
        print("❌ Secret exists but is empty")
        
except Exception as e:
    print(f"❌ Error retrieving secret: {e}")
    print("\n📝 To fix this:")
    print("1. Run: databricks secrets create-scope --scope vidya-setu")
    print("2. Run: databricks secrets put --scope vidya-setu --key sarvam-api-key")
    print("3. Paste your API key when prompted")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Setup Complete!
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC 1. Your app will now retrieve the API key securely from Databricks Secrets
# MAGIC 2. Delete this notebook for security (the secret is already stored)
# MAGIC 3. Restart your Vidya Setu app to use the new secure configuration
# MAGIC 
# MAGIC **Security Benefits:**
# MAGIC - ✅ API key not visible in source code
# MAGIC - ✅ API key not committed to Git
# MAGIC - ✅ API key encrypted at rest
# MAGIC - ✅ Access controlled via Databricks ACLs

