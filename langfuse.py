import os
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

# Get keys for your project from the project settings page
# https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]

# Your host, defaults to https://cloud.langfuse.com
# For US data region, set to "https://us.cloud.langfuse.com"
# os.environ["LANGFUSE_HOST"] = "http://localhost:3000"


langfuse = Langfuse()
