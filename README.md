# Language Evaluator

This project is a **language evaluator tool** built with Python and Streamlit.It allows users to upload a CSV file containing reference and prediction texts and calculates BLEU scores for each row. The tool supports BLEU score evaluation in both **English** and **Korean** text.

## Features
- Supports BLEU score evaluation for English and Korean.
- Displays BLEU scores for each row in the CSV file.
- Calculates and displays the average BLEU score.
- Easy-to-use web interface built with Streamlit.

## Tokenization
- For English text, it uses NLTK's Word Tokenize.
- For Korean text, it uses KoNLPy’s Okt tokenizer. <br>
(**Java** must be installed on your system to use KoNLPy’s Okt tokenizer)

## How to Start the Project  
**Note**: This project was developed using **Python 3.11**.

1. Install the required Python libraries:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run main.py
```
