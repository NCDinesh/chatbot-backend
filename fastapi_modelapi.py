from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import joblib
import os
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the base model with quantization
base_model = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained("ncdinesh2057/mistral_genderbasedviolencefinetuned_qlora")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quantization_config,
    device_map={"": 0}
)

# Load fine-tuned PEFT model
model = PeftModel.from_pretrained(model, "ncdinesh2057/mistral_genderbasedviolencefinetuned_qlora")
model = model.to(device)

# Define prompt template
template = """
<s>[INST] <<SYS>>
You are a helpful AI assistant who has to act as an advocate in the case of country Nepal, not others.
Answer based on the context provided. Don't answer unnecessarily if you don't find the context.
<</SYS>>
{context}
Question: {question}
Helpful Answer: [/INST]
"""

prompt = PromptTemplate.from_template(template)

# Load PDF and process it
reader = PdfReader('data/fine_tune_data.pdf')
raw_text = ''
for page in reader.pages:
    text = page.extract_text()
    if text:
        raw_text += text

# Split text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=350,
    chunk_overlap=20,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Load or create embeddings
embeddings_file = "./data/genderviolence.joblib"
if os.path.exists(embeddings_file):
    embeddings = joblib.load(embeddings_file)
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    joblib.dump(embeddings, embeddings_file)

vectorstore = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()

def generate_response(prompt_text, max_new_tokens=500, temperature=0.7):
    """Generate response using the locally loaded Mistral model."""
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
            temperature=temperature,
            do_sample=True,
            top_p=0.95
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)



# FastAPI Request Model
class QueryRequest(BaseModel):
    query: str
    max_new_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    context_length: Optional[int] = 2048


# Initialize FastAPI app
app = FastAPI()

@app.post("/query/")
async def query_endpoint(request: QueryRequest):
    query = request.query

    if query:
        start_time = time.time()

        # Retrieve relevant documents
        documents = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in documents])

        # Format the prompt
        full_prompt = template.format(context=context, question=query)

        # Generate response locally
        response_text = generate_response(full_prompt, request.max_new_tokens, request.temperature)

        # Extract response after [/INST]
        response_start = response_text.find('[/INST]')
        if response_start != -1:
            response_after_inst = response_text[response_start + len('[/INST]'):].strip()
        else:
            response_after_inst = "Sorry, no valid answer generated."

        end_time = time.time()

        return {
            "response": response_after_inst,
            "response_time": f"{end_time - start_time:.2f} seconds"
        }
    else:
        raise HTTPException(status_code=400, detail="Please enter a question.")


# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
