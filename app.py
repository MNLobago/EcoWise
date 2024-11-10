import os
import sys
import gc
import psutil
import gradio as gr
import keras_nlp
from huggingface_hub import login

# Check if the API key is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python app.py <your_hugging_face_api_key>")
    sys.exit(1)

# Get the API key from command-line arguments
api_key = sys.argv[1]

# Log in with the provided Hugging Face API token
try:
    login(api_key)
except Exception as e:
    print(f"Login failed: {e}")
    sys.exit(1)

# Load the Keras NLP model from Hugging Face
model_path = "MNLobago/EcoWise_model"
try:
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(f"hf://{model_path}")
    print(f"Model {model_path} loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def get_memory_usage():
    """Return memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # in MB

class GemmaChat:
    def __init__(self, model, max_length=150, system=""):
        self.model = model
        self.max_length = max_length
        self.system = system
        self.history = []  # Reset history handling for simplicity

    def get_full_prompt(self, user_input):
        """Constructs the complete prompt for the model based on the current user input."""
        # Constructing a clean prompt that only includes the user input
        prompt = f"User: {user_input}\nModel:"
        return prompt

    def query(self, question):
        """Generates a response from the model based on the user's question."""
        # Only use the system message for the first interaction
        if not self.history:
            prompt = self.system + "\n" + self.get_full_prompt(question) if self.system else self.get_full_prompt(question)
        else:
            prompt = self.get_full_prompt(question)
        
        print(f"Prompt sent to model: {prompt}")  # Debugging step
        
        try:
            # Generate model response
            response = self.model.generate(prompt, max_length=self.max_length)
            print("Raw Model Response:", response)
            
            # Clean and trim the model response
            model_response = response.replace(prompt, "").strip()
            
            # Sanitize the response to ensure it doesn't contain a question or the prompt
            if model_response.endswith('?'):
                model_response = model_response.rstrip('?') + '.'  # Remove trailing question marks
            print(f"Trimmed Model Response: {model_response}")
        except Exception as e:
            print("Error generating response:", e)
            return "I'm sorry, there was an error generating a response."

        gc.collect()  # Manual garbage collection to manage memory

        return model_response

# Initialize the chat object with a more focused system prompt
chat = GemmaChat(
    model=gemma_lm, 
    system="""You are an intelligent chatbot focused on answering questions related to climate change, sustainability, and carbon footprint."""
)

def chat_with_model(input_text):
    """Handles chat interactions."""
    # Reset history to avoid feedback loops
    chat.history = []

    # Query the model and return the formatted response
    answer = chat.query(input_text)

    # Return response formatted correctly for Gradio as a list of tuples
    return [
        ("user", input_text),  # User's message as tuple
        ("model", answer)      # Model's response as tuple
    ]

# Create and launch the Gradio interface
demo = gr.Interface(
    fn=chat_with_model,
    inputs="text",
    outputs="chatbot",
    description="🌍 Welcome to EcoWise, your go-to climate-savvy chatbot! I'm here to help you."
)

# Launch the demo
demo.launch(share=True, debug=True)
