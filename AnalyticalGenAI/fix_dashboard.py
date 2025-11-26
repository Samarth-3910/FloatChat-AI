# Quick fix script for oceanography_dashboard.py
import re

# Read the file
with open('oceanography_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Change database name from SIH to floatchatAI
content = content.replace("db_name = 'SIH'", "db_name = 'floatchatAI'")

# Fix 2: Add None check after load_data()
# Find the main() function and add the check
old_pattern = r"(def main\(\):.*?df = load_data\(\)\s+# Header)"
new_text = r"\1\n    \n    # Check if data loaded successfully\n    if df is None or df.empty:\n        st.error(\"❌ Unable to load data. Please check your database connection.\")\n        st.info(\"💡 Make sure PostgreSQL is running and the 'floatchatAI' database contains the 'sample_gold_layer' table.\")\n        return\n    \n    # Header"

content = re.sub(old_pattern, new_text, content, flags=re.DOTALL)
 
# Write back
with open('oceanography_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed database name and added None check!")
