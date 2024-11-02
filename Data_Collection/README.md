# Data Collection Preprocessing for Carbon Footprint Q&A

This repository contains a comprehensive data collection and preprocessing pipeline designed to build a dataset of questions and answers related to carbon footprint and climate change. The data has been gathered from various reputable websites, processed, and enriched for further training and analysis.

## Overview

The primary aim of this project is to collect questions and answers about climate change and carbon footprint, preprocess the data, and generate additional Q&A pairs for enhanced diversity and coverage. The preprocessing leads to two main outputs: a PDF and a JSON file containing structured Q&A data.

## Data Sources

The data was collected from the following websites:

1. [EESI - Climate Change FAQ](https://www.eesi.org/climate-change-FAQ)
2. [WRI - 6 Pressing Questions About Beef and Climate Change](https://www.wri.org/insights/6-pressing-questions-about-beef-and-climate-change-answered)
3. [WWF - Common Climate Change Questions](https://www.wwf.org.uk/updates/we-answer-your-most-common-climate-change-questions)
4. [Cheshire West and Chester - Climate Emergency FAQs](https://participatenow.cheshirewestandchester.gov.uk/climate-emergency/widgets/28517/faqs)
5. [Nature Australia - Climate Change FAQs](https://www.natureaustralia.org.au/what-we-do/our-priorities/climate-change/climate-change-stories/climate-change-frequently-asked-questions/)
6. [IMF - Climate Change FAQs](https://www.elibrary.imf.org/view/journals/026/2015/004/article-A001-en.xml)
7. [ResearchGate - Carbon Emission](https://www.researchgate.net/topic/Carbon-Emission)
8. [BAFU - Climate Change Questions & Answers](https://www.bafu.admin.ch/bafu/en/home/topics/climate/questions-answers.html)
9. [EcoAct - Calculate a Carbon Footprint for Your Business](https://eco-act.com/blog/how-to-calculate-a-carbon-footprint-for-your-business/)
10. [EESI (Duplicate)](https://www.eesi.org/climate-change-FAQ)
11. [CSIRO - Climate Change Q&A](https://www.csiro.au/en/research/environmental-impacts/climate-change/climate-change-qa)
12. [EPA - FAQs About Climate Change](https://www.epa.gov/climatechange-science/frequently-asked-questions-about-climate-change)
13. [ResearchGate - Carbon Footprint](https://www.researchgate.net/topic/Carbon-Footprint)

## Data Extraction and Processing

The collection and extraction of data were done using the following tools:

1. **[Firecrawl.dev](https://firecrawl.dev)**: This tool was used to convert the web pages into Markdown format.
2. **[DeepAI ChatGPT](https://deepai.org/chat/free-chatgpt)**: The extracted Markdown content was then processed using prompt engineering to extract relevant questions and answers.

### Files

In the repository, you will find several text files containing the exported Markdown data from the various sites:
- `new.txt`, `new1.txt`, `new2.txt`, ..., `new10.txt`: These files contain the Markdown content from various sources.

### Jupyter Notebooks

1. **DataGenrator.ipynb**: This notebook generates the unprocessed data from the Markdown files.
  
2. **DataGenrator_PREPROCESSED.ipynb**: This notebook processes the data to create well-defined Q&A pairs. Key steps include merging lists, generating variations using synonyms, and creating output files in different formats (PDF and JSON).

### Key Features of `DataGenrator_PREPROCESSED.ipynb`

- Merging existing question and answer lists while retaining their order.
- Checking for unique questions and answers extracted, ensuring diversity in training data.
- Generating additional Q&A pairs using synonyms to enrich the dataset.
- Outputting the results into a PDF file (`high_volume_carbon_footprint_qa.pdf`) and a JSON file (`high_volume_carbon_footprint_qa.json`).

### Python Functions

- **replace_terms**: Replaces key terms in the text with synonyms.
- **generate_additional_qa**: Generates additional unique Q&A entries.
- **create_pdf**: Creates a PDF file containing Q&A data.
- **create_json**: Exports the Q&A data into a JSON format.

## Results

- **Initial Data Counts**:
  - Total initial questions: 796
  - Total unique questions generated: 3295
- **Total Q&A Pairs**: 45,863

## Usage

To use the notebooks:

1. Ensure all required libraries (like `pandas`, `reportlab`, etc.) are installed.
2. Open the notebooks in Jupyter Notebook or JupyterLab.
3. Execute the cells sequentially to perform data generation and preprocessing.
4. Review the generated PDF and JSON files for the complete dataset.