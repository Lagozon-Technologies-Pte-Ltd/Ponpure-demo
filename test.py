import json
import os
import warnings
import pyodbc
from openai import AzureOpenAI
from dotenv import load_dotenv
import yaml

load_dotenv()
warnings.filterwarnings("ignore")

# ---------------------- Azure OpenAI Client ----------------------
azure_openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")

# ---------------------- SQL Server Connection ----------------------
conn = pyodbc.connect(
    "DRIVER=ODBC Driver 18 for SQL Server;"
    "SERVER=lzgenaipoc.database.windows.net,1433;"
    "DATABASE=lzgenaipocdb;"
    "UID=sqladmin;"
    "PWD=Xj3DvuM#;"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
)

cursor = conn.cursor()

# ---------------------- Filter Tables ----------------------
TARGET_TABLES = [
    "Ponpure_DispatchDetails",
    "Ponpure_LeadDetails",
    "Ponpure_Quotationhdrs",
    "Ponpure_SaleOrderdtls",
    "Ponpure_SaleorderHdrs",
    "Ponpure_Schedules"
]

# ---------------------- Azure OpenAI Description Generator ----------------------
def ai_generate_description(table, column, dtype, is_pk):
    prompt = f"""
Write a short (1–2 sentence) business-friendly description for this column:

Table: {table}
Column: {column}
Data Type: {dtype}
Primary Key?: {is_pk}

Write it as if for a professional data dictionary.
"""
    try:
        response = azure_openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except:
        return f"Auto-description for {column} in {table}."

# ---------------------- Schema Helpers ----------------------
def get_columns(table):
    cursor.execute(f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table}'
    """)
    return cursor.fetchall()

def get_primary_keys(table):
    cursor.execute(f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_NAME='{table}'
        AND CONSTRAINT_NAME LIKE 'PK_%'
    """)
    return [row[0] for row in cursor.fetchall()]

def get_foreign_keys(table):
    cursor.execute(f"""
        SELECT 
            CU.COLUMN_NAME,
            PK.TABLE_NAME AS REFERENCED_TABLE,
            PT.COLUMN_NAME AS REFERENCED_COLUMN
        FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS RC
        JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE CU
            ON CU.CONSTRAINT_NAME = RC.CONSTRAINT_NAME
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE PT
            ON PT.CONSTRAINT_NAME = RC.UNIQUE_CONSTRAINT_NAME
        JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS PK
            ON PK.CONSTRAINT_NAME = RC.UNIQUE_CONSTRAINT_NAME
        WHERE CU.TABLE_NAME='{table}'
    """)
    return cursor.fetchall()

def get_example_values(table, column):
    try:
        cursor.execute(f"SELECT TOP 2 [{column}] FROM {table} WHERE [{column}] IS NOT NULL")
        rows = cursor.fetchall()
        return [str(r[0]) for r in rows]
    except:
        return []

# ---------------------- Build Column Metadata ----------------------
column_json = []
all_columns = {}
pk_map = {}

# Collect schema for the 6 tables
for table in TARGET_TABLES:
    cols = get_columns(table)
    pks = get_primary_keys(table)
    pk_map[table] = pks
    all_columns[table] = [c[0] for c in cols]

# Detect join possibilities
def detect_joins(table, col):
    joins = []
    for other in TARGET_TABLES:
        if other == table:
            continue
        if col in all_columns[other] or col in pk_map.get(other, []):
            joins.append({
                "table": other,
                "on": col,
                "join_type": "LEFT OUTER JOIN"
            })
    return joins

# Build metadata
for table in TARGET_TABLES:
    cols = get_columns(table)
    pks = pk_map[table]

    for col_name, dtype, nullable in cols:
        is_pk = col_name in pks
        desc = ai_generate_description(table, col_name, dtype, is_pk)
        examples = get_example_values(table, col_name)

        column_json.append({
            "column_name": f"{table}.{col_name}",
            "column_desc": desc,
            "metadata": {
                "type": "column",
                "table_name": table,
                "data_type": dtype.upper(),
                "nullable": (nullable == "YES"),
                "is_primary_key": is_pk,
                "is_foreign_key": False,
                "joins": detect_joins(table, col_name)
            },
            "examples": examples
        })

# Save metadata_output.json
with open("column_metadata.json", "w", encoding="utf-8") as f:
    json.dump(column_json, f, indent=2, ensure_ascii=False)

print("Created → column_metadata.json")

# ---------------------- Table Metadata ----------------------
table_json = []
for table in TARGET_TABLES:
    table_json.append({
        "id": table,
        "document": f"{table} table — contains important business data.",
        "metadata": {
            "type": "table",
            "primary_key": pk_map.get(table, []),
            "join_guidance": []
        }
    })

with open("table_metadata.json", "w", encoding="utf-8") as f:
    json.dump(table_json, f, indent=2, ensure_ascii=False)

print("Created → table_metadata.json")

# ---------------------- Relationship YAML ----------------------
relationships = []

for table in TARGET_TABLES:
    fks = get_foreign_keys(table)
    for col, ref_table, ref_col in fks:
        if ref_table in TARGET_TABLES:
            relationships.append({
                "left_table": table,
                "left_columns": [col],
                "right_table": ref_table,
                "right_columns": [ref_col],
                "cardinality": "ManyToOne",
                "description": f"{table} → {ref_table} via {col}"
            })

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

yaml_output = {"RELATIONSHIPS": relationships}

with open("table_relationship.yaml", "w", encoding="utf-8") as f:
    yaml.dump(yaml_output, f, sort_keys=False, allow_unicode=True, Dumper=NoAliasDumper)

print("Created → table_relationship.yaml")
