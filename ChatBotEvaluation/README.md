# ChatBot Evaluation Repository

This repository contains the code and data used for evaluating various chatbot models. The evaluation process includes both automated scoring (using scripts with human feedback) and human feedback.

## Evaluation Process Flow
1. **Human Evaluation**: 
   - Begin with **`human-chatbot-evaluation.ipynb`**. This notebook outlines how questions were selected and sent via Google Forms to various participants (community, classmates, and friends). It provides details about the human evaluation process.
   
2. **Data Collection**: 
   - Use **`chatbotevaluation.ipynb`**. This notebook describes the methodology for collecting data to evaluate the chatbot and links the collected data to the performance metrics.
   
3. **Scoring**: 
   - Conclude with **`ScoringINF.ipynb`**. This notebook is designed to score the performance of different chatbot models. It makes use of the custom script **`model_scoring.py`**, which handles the scoring for all model versions.

## Contents

### Notebooks
- **`human-chatbot-evaluation.ipynb`**: Describes the process of selecting questions and conducting human evaluations.
- **`chatbotevaluation.ipynb`**: Methodology for collecting evaluation data and establishing performance metrics.
- **`ScoringINF.ipynb`**: Scores the performance of different chatbot models using `model_scoring.py`.

### Scripts
- **`model_scoring.py`**: A Python script used in the `ScoringINF.ipynb` notebook for scoring the different chatbot models based on performance metrics.

### Data Files
- **`model_responses_with_time_tqdm/*.csv`**: Contains the model responses along with evaluation scores collected during the evaluation phase.
- **`model_responses_with_time_tqdm*S*.csv`**: A variant of the model responses CSV, potentially containing additional or refined data.
- **`model_responses_with_time_tqdm*S*.xlsx`**: An Excel version of the evaluation scores and model responses for easy access and analysis.

### Performance Metrics
- **`performance_metrics_by_category.png`**: A graphical representation of performance metrics by category, summarizing the evaluation results visually.

### Model Folders
We have three folders named **Model1**, **Model2**, and **Model3**, each containing similar content for easy comparison between different chatbot models.

### LLM Evaluation
- **LLM_Evaluation/**: A newly added folder containing files related to the evaluation of the three model variants using a language model (LLM). This folder includes methodologies and results pertaining to the LLM's performance comparison against human evaluation.

## Data Sources
Here are the sources of the data collected for the quiz:

- [CLIMATE LITERACY QUIZ](https://cleanet.org/clean/literacy/climate/quiz.html)
- [Quizizz: Carbon Footprint Quiz](https://quizizz.com/admin/quiz/5abd06abd0f9b800198aa328/carbon-footprint)
- [Online Exam Maker: Climate Change Quiz Questions and Answers](https://onlineexammaker.com/kb/30-climate-change-quiz-questions-and-answers/)