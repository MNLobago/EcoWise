# ChatBot Evaluation Repository

This repository contains the code and data used for evaluating various chatbot models. The evaluation process includes both automated scoring (Using Script With Human FeedBack) and human feedback.

## Contents

### Notebooks
- **ScoringINF.ipynb**: A notebook for scoring the performance of different chatbot models.
- **human-chatbot-evaluation.ipynb**: This notebook outlines how questions were selected and sent via Google Forms to my community, classmates, and friends. It provides details about the human evaluation process.
- **chatbotevaluation.ipynb**: This notebook describes the methodology for collecting data to evaluate the chatbot and links the collected data to the performance metrics.

### Scripts
- **model_scoring.py**: A Python script used by the `chatbotevaluation.ipynb` notebook for scoring the different chatbot models based on the performance metrics obtained.

### Data Files
- **model_responses_with_time_tqdm\*.csv**: Contains the model responses along with evaluation scores collected during the evaluation phase.
- **model_responses_with_time_tqdm\*S\*.csv**: A variant of the model responses CSV, potentially containing additional or refined data.
- **model_responses_with_time_tqdm\*S\*.xlsx**: An Excel version of the evaluation scores and model responses for easy access and analysis.

### Performance Metrics
- **performance_metrics_by_category.png**: A graphical representation of performance metrics by category, summarizing the evaluation results visually.

### Model Folders
We have three folders named **Model1**, **Model2**, and **Model3**, each containing the same content mentioned above. These folders allow for easy comparison between the different chatbot models.

## Data Sources
Here are the sources of the data collected for the quiz:

- [CLIMATE LITERACY QUIZ](https://cleanet.org/clean/literacy/climate/quiz.html)
- [Quizizz: Carbon Footprint Quiz](https://quizizz.com/admin/quiz/5abd06abd0f9b800198aa328/carbon-footprint)
- [Online Exam Maker: Climate Change Quiz Questions and Answers](https://onlineexammaker.com/kb/30-climate-change-quiz-questions-and-answers/)
