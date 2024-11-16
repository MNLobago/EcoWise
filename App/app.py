import os
import gc
import psutil
import gradio as gr
import keras_nlp
from huggingface_hub import login

# Get the API key from environment variable
api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    raise ValueError("Please set the 'HUGGINGFACE_API_KEY' environment variable.")

# Log in with the provided Hugging Face API token
login(api_key)

# Load the Keras NLP model from Hugging Face
model_path = "MNLobago/EcoWise_model"
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(f"hf://{model_path}")

class GemmaChat:
    def __init__(self, model, max_length=150, system=""):
        self.model = model
        self.max_length = max_length
        self.system = system
        self.history = []

    def get_full_prompt(self, user_input):
        return f"User: {user_input}\nModel:"

    def query(self, question):
        if not self.history:
            prompt = self.system + "\n" + self.get_full_prompt(question) if self.system else self.get_full_prompt(question)
        else:
            prompt = self.get_full_prompt(question)
        
        response = self.model.generate(prompt, max_length=self.max_length)
        model_response = response.replace(prompt, "").strip()
        
        # Sanitize the response
        if model_response.endswith('?'):
            model_response = model_response.rstrip('?') + '.'

        gc.collect()
        return model_response

# Initialize the chat object
chat = GemmaChat(
    model=gemma_lm,
    system="""You are an intelligent chatbot focused on answering questions related to climate change, sustainability, and carbon footprint."""
)

def chat_with_model(input_text):
    chat.history = []
    answer = chat.query(input_text)
    return [("user", input_text), ("model", answer)]

# Create and launch the Gradio interface
demo = gr.Interface(
    fn=chat_with_model,
    inputs="text",
    outputs="chatbot",
    description="🌍 Welcome to EcoWise, your go-to climate-savvy chatbot! I'm here to help you."
)

demo.launch()