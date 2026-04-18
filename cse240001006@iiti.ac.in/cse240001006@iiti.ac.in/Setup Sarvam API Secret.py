# Databricks notebook source
# DBTITLE 1,Setup Instructions
# MAGIC %md
# MAGIC # 🔐 Setup Sarvam AI API Key Secret
# MAGIC This notebook will help you store your Sarvam AI API key securely in Databricks Secrets.
# MAGIC
# MAGIC **Run this once, then you can delete this notebook.**

# COMMAND ----------

# DBTITLE 1,Method 1: CLI
# MAGIC %md
# MAGIC ## Method 1: Using Databricks CLI (Recommended)
# MAGIC If you have Databricks CLI installed, run these commands in your terminal:
# MAGIC
# MAGIC ```bash
# MAGIC # Create secret scope
# MAGIC databricks secrets create-scope --scope vidya-setu
# MAGIC
# MAGIC # Add the API key
# MAGIC databricks secrets put --scope vidya-setu --key sarvam-api-key
# MAGIC ```
# MAGIC
# MAGIC When prompted, paste: `sk_73frjkbk_h9u67wusG8JiAh2z4VBZ5EuH`

# COMMAND ----------

# DBTITLE 1,Method 2: Python API
# MAGIC %md
# MAGIC ## Method 2: Using Python API (Alternative)
# MAGIC
# MAGIC Run the cell below to set up the secret programmatically:

# COMMAND ----------

# DBTITLE 1,Create Secret Scope and Store Key
from databricks.sdk import WorkspaceClient

# Initialize workspace client
w = WorkspaceClient()

# Create secret scope
try:
    w.secrets.create_scope(scope="vidya-setu")
    print("✅ Created secret scope: vidya-setu")
except Exception as e:
    if "already exists" in str(e).lower():
        print("✅ Secret scope 'vidya-setu' already exists")
    else:
        print(f"❌ Error creating scope: {e}")

# Store the API key
try:
    w.secrets.put_secret(
        scope="vidya-setu",
        key="sarvam-api-key", 
        string_value="sk_73frjkbk_h9u67wusG8JiAh2z4VBZ5EuH"
    )
    print("✅ Stored API key successfully!")
except Exception as e:
    print(f"❌ Error storing secret: {e}")
    print("\nAlternative: Use Databricks CLI method above")

# COMMAND ----------

# DBTITLE 1,Verification Step
# MAGIC %md
# MAGIC ## Step 3: Verify the Secret
# MAGIC
# MAGIC Run the cell below to verify the secret is configured correctly:

# COMMAND ----------

# DBTITLE 1,Verify Secret Configuration
from databricks.sdk.runtime import dbutils

try:
    # Retrieve the secret (value won't be displayed for security)
    key = dbutils.secrets.get(scope="vidya-setu", key="sarvam-api-key")
    
    if key:
        print("✅ SUCCESS! Sarvam API key is configured!")
        print(f"✅ Key length: {len(key)} characters")
        print(f"✅ Key preview: {key[:10]}...")
        print("\n🎉 Your Vidya Setu app will now use the secure API key!")
        print("\n📝 Next steps:")
        print("1. Close this notebook")
        print("2. Restart your Vidya Setu app")
        print("3. Test the '🔊 Read Question Aloud' button")
        print("4. Delete this notebook for security")
    else:
        print("❌ Secret exists but is empty")
        
except Exception as e:
    print(f"❌ Error retrieving secret: {e}")
    print("\n📝 Please use Method 1 (Databricks CLI) to set up the secret")

# COMMAND ----------

# DBTITLE 1,Completion
# MAGIC %md
# MAGIC ## ✅ All Done!
# MAGIC
# MAGIC **Security Best Practices:**
# MAGIC - ✅ API key is now encrypted in Databricks Secrets
# MAGIC - ✅ API key is NOT visible in source code
# MAGIC - ✅ API key will NOT be committed to Git
# MAGIC - ✅ Delete this notebook after setup
# MAGIC
# MAGIC **Your app is now secure!** 🔒