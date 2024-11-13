# MNLobago Repository

Welcome to the MNLobago repository, which houses the EcoWise chatbot—a project designed to provide users with valuable insights and information on environmental sustainability and climate change. The EcoWise chatbot aims to educate and engage users on pressing environmental issues, promoting awareness and fostering sustainable practices.

## Repository Structure

1. **EcoWise**  
   Code and materials related to the EcoWise chatbot.

2. **ChatBotEvaluation**  
   Scripts and documentation for assessing the chatbot's performance and effectiveness.

3. **Data_Collection_Creation**  
   Tools and resources for collecting and creating datasets for training and fine-tuning the chatbot.

4. **Data_Formats_CSVs**  
   Formatted CSV files ready for use in Phase 1 of the project.

5. **Data_Quality**  
   Documentation of data quality checks to ensure high-quality datasets.

6. **Gemma_ModelFineTuning**  
   Scripts for fine-tuning the Gemma-2B language model for environmental topic responses.

7. **Project Workflow**  
   A diagram illustrating the project's workflow, updated on 2024-11-05.

8. **app.py**  
   The main deployment script for the EcoWise chatbot, enabling user interaction with the model via an interface.

## How to Use `app.py`

`app.py` is the main script for running the EcoWise chatbot, providing an interactive interface for users.

### Prerequisites

Before you start, make sure you have:

- **Python 3.6 or higher**: Ensure Python is installed on your machine. Download it from [python.org](https://www.python.org/downloads/).
- **Required Libraries**: Install the necessary Python libraries by running:

  ```bash
  pip install gradio keras-nlp huggingface-hub psutil
  ```

- **Hugging Face API Key**: Sign up at [Hugging Face](https://huggingface.co/join) and obtain your API token.

### Execute the Script

Run the `app.py` script using Python, supplying your Hugging Face API key as an argument:

```bash
python app.py <Hugging_face_api_key>
```

Replace `Hugging_face_api_key>` with your actual API key.

### Interact with the Chatbot

Upon successful execution, an interactive Gradio interface will open in your default web browser. You can start chatting with the EcoWise chatbot by typing your questions related to environmental sustainability and climate change.

### Example Command

```bash
python app.py API_KEY_HERE
```