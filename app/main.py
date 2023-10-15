from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

# Declare data object with the components and types
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int

# Declare a simple data object
class Value(BaseModel):
    value: int

# Instantiate the app
app = FastAPI()

# Example data input
path = "shoes"
query = 3
body = "hello"

# Define a POST on the specified endpoint
@app.post("/items/")
async def create_item(item: TaggedItem):
    return item

# Define a GET method for item_id
@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int = 1):
    return {"fetch": f"Fetched {count} of {item_id}"}

@app.post("/{path}")
async def exercise_function(path: int, query: int, body: Value):
    return {"path": path, "query": query, "body": body}