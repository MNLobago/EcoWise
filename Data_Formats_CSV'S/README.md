# Make_CSV: JSON to CSV Data Format Transformation

`Make_CSV` is a Jupyter Notebook that processes a JSON file containing questions and answers about carbon footprint data. The notebook generates two different CSV files (`format1.csv` and `format2.csv`) in distinct formats to cater to various use cases.

## Overview

The primary objective of this notebook is to convert the given JSON file (`high_volume_carbon_footprint_qa.json`) into two structured CSV formats using Python and the Pandas library.

### Formats Generated

1. **Format 1 (`format1.csv`)**: 
   - Contains the columns:
     - `question`
     - `answer`
   - This format is straightforward and ideal for tasks focused on direct question-answer retrieval.

2. **Format 2 (`format2.csv`)**:
   - Contains the columns:
     - `id`
     - `qid`
     - `docid`
     - `context` (previously `answer`)
     - `question`
   - This more structured format includes unique identifiers for each question and is suited for use cases that require metadata along with the text.

## Requirements

To run the notebook, ensure you have the following packages installed:

- `pandas`
- `json`

You can install any missing packages using pip:

```bash
pip install pandas