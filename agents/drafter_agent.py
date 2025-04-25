# agents/drafter_agent.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFacePipeline

# Detect device
device = "cpu"
print(f"Device set to use {device}")

# Load tokenizer and model
model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create text-generation pipeline with higher max tokens
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_new_tokens=300,  # Limit *just the output length*
    do_sample=True,
    temperature=0.7,
    device=0 
)



# Wrap with LangChain-compatible LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["research_summary"],
    template="""
You are an AI answer drafter. Based on the structured research summary below, generate a professional, clear, and concise answer. Include citations if available.

Research Summary:
{research_summary}

Drafted Answer:"""
)

# Build LangChain Runnable
drafter_chain = prompt_template | llm

# Format post-processing to extract just the answer
def clean_drafted_output(text: str) -> str:
    last_period_index = text.rstrip().rfind(". ")
    if last_period_index != -1:
        return text[:last_period_index + 1].strip()
    return text.strip()

# Function to generate draft
def generate_draft(research_summary: str) -> str:
    raw_output = drafter_chain.invoke({"research_summary": research_summary})
    drafted_text = raw_output["text"] if isinstance(raw_output, dict) else raw_output
    cleaned_output = clean_drafted_output(drafted_text)
    return cleaned_output