# # src/BOT/llm_groq.py
# import os
# import boto3
# from decimal import Decimal
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # ---------------- Groq API key fallback ---------------- #
# GROQ_API_KEY = "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"

# # ---------------- DynamoDB setup ---------------- #
# dynamodb = boto3.resource("dynamodb", region_name="us-west-2") 
#  # Specify your region
# MODEL_CONFIG_TABLE_NAME = "model_config_table"
# model_config_table = dynamodb.Table(MODEL_CONFIG_TABLE_NAME)

# # ---------------- Helpers ---------------- #
# def from_dynamo(obj):
#     """Recursively convert DynamoDB Decimals to float for JSON compatibility"""
#     if isinstance(obj, dict):
#         return {k: from_dynamo(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [from_dynamo(v) for v in obj]
#     if isinstance(obj, Decimal):
#         return float(obj)
#     return obj

# def get_llm_config(pk: str = "sharepoint"):
#     """
#     Fetch the main_model configuration from DynamoDB
#     Returns a dict with keys: model, api_key, temperature, max_tokens
#     """
#     try:
#         response = model_config_table.get_item(Key={"pk": pk})
#         item = response.get("Item")
#         if not item:
#             print(f"No config found for pk={pk}, using environment fallback")
#             return None
#         main_model = item.get("main_model")
#         if not main_model:
#             print(f"No main_model for pk={pk}, using environment fallback")
#             return None
#         return from_dynamo(main_model)
#     except Exception as e:
#         print(f"Error fetching main_model from DynamoDB: {e}")
#         return None

# # ---------------- Initialize LLM ---------------- #
# llm_config = get_llm_config("sharepoint")

# if llm_config:
#     print(f"Initializing LLM from DynamoDB config: {llm_config}")
#     llm = ChatGroq(
#         model=llm_config.get("model"),
#         api_key=GROQ_API_KEY,
#         temperature=llm_config.get("temperature", 0.3),
#         max_tokens=int(llm_config.get("max_tokens", 100)),
#     )
# else:
#     print("Initializing LLM from environment fallback")
#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         api_key=GROQ_API_KEY,
#         temperature=0.3,
#         max_tokens=100,
#     )

# print(f"LLM initialized with model: {llm_config.get('model', 'llama-3.3-70b-versatile') if llm_config else 'llama-3.3-70b-versatile'}")





import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",  # Or "llama3-70b-8192", etc.
    api_key=GROQ_API_KEY,
    temperature=0.3,
    max_tokens=100,
)
