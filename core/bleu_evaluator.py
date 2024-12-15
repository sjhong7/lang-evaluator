from typing import List
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from konlpy.tag import Okt

nltk.download('punkt')
nltk.download('punkt_tab')
okt = Okt()

def calculate_bleu_scores(
    df: pd.DataFrame, 
    reference_col: str = "reference", 
    prediction_col: str = "prediction", 
    lang: str = "en"
) -> List[float]:
    """
    Calculates BLEU scores for each row in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing reference and prediction columns.
        reference_col (str): Column name for reference text.
        prediction_col (str): Column name for predicted text.
        lang (str): Language code ('en' or 'ko').

    Returns:
        List[float]: A list of BLEU scores.
    """
    smooth = SmoothingFunction().method4
    bleu_scores = []

    for _, row in df.iterrows():
        reference_text = row[reference_col]
        prediction_text = row[prediction_col]

        # Using different tokenizers depends on the chosen language
        if lang == 'en':
            reference_tokens = nltk.word_tokenize(reference_text)
            prediction_tokens = nltk.word_tokenize(prediction_text)
        elif lang == 'ko':
            reference_tokens = okt.morphs(reference_text)
            prediction_tokens = okt.morphs(prediction_text)
        else:
            raise ValueError("Unsupported language. Use 'en' for English or 'ko' for Korean.")

        # calculate BLEU scores
        bleu_score = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu_score)

    return bleu_scores
