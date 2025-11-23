"""
Response validation utilities for detecting gibberish and repetitive content

Used for transparent reporting of model failures without cherry-picking.
Reports stratified metrics: overall, valid-only, and failure rates.
"""

import re
from typing import Dict


def detect_gibberish(text: str) -> bool:
    """
    Detect gibberish characters in text

    Checks for:
    - Unicode replacement characters
    - Null bytes
    - High concentration of non-ASCII characters
    - Repeated special characters

    Args:
        text: Response text to check

    Returns:
        True if gibberish detected
    """
    # Handle non-string inputs (e.g., NaN from CSV)
    if not isinstance(text, str):
        return True
    if not text or len(text.strip()) == 0:
        return True

    # Check for Unicode gibberish characters
    gibberish_chars = ['�', '\ufffd', '\x00', '\uf8ff']
    if any(char in text for char in gibberish_chars):
        return True

    # Check for high concentration of repeated special characters
    # e.g., "ʂʂʂʂʂ" or "████"
    special_pattern = r'(.)\1{10,}'  # Same character repeated 10+ times
    if re.search(special_pattern, text):
        return True

    # Check for mostly non-printable or unusual Unicode
    printable_ratio = sum(1 for c in text if c.isprintable() or c.isspace()) / len(text)
    if printable_ratio < 0.8:
        return True

    return False


def detect_repetition(text: str, threshold: float = 0.3) -> bool:
    """
    Detect repetitive content in text

    Checks if >threshold of text is repeated sentences/phrases.

    Args:
        text: Response text to check
        threshold: Fraction of repetition to flag (default 0.3 = 30%)

    Returns:
        True if repetitive content detected
    """
    # Handle non-string inputs (e.g., NaN from CSV)
    if not isinstance(text, str):
        return True
    if not text or len(text.strip()) == 0:
        return True

    # Check for same word repeated 3+ times (with or without separator)
    # e.g., "#coronavirus#coronavirus#coronavirus" or "word word word"
    word_repeat = r'(\b\w{3,}\b).*?\1.*?\1'
    if re.search(word_repeat, text.lower()):
        # Count how many times the most repeated word appears
        words = re.findall(r'\b\w{3,}\b', text.lower())
        if words:
            from collections import Counter
            word_counts = Counter(words)
            most_common_count = word_counts.most_common(1)[0][1]
            # Flag if any word appears more than 5 times
            if most_common_count > 5:
                return True

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

    if len(sentences) <= 1:
        return False

    # Check sentence-level repetition
    unique_sentences = set(sentences)
    repetition_ratio = 1 - (len(unique_sentences) / len(sentences))

    if repetition_ratio > threshold:
        return True

    # Check for repeated phrases (n-grams)
    words = text.lower().split()
    if len(words) >= 10:
        # Check for repeated 5-grams
        ngrams = [' '.join(words[i:i+5]) for i in range(len(words)-4)]
        unique_ngrams = set(ngrams)
        ngram_repetition = 1 - (len(unique_ngrams) / len(ngrams))

        if ngram_repetition > threshold:
            return True

    return False


def validate_response(text: str) -> Dict[str, bool]:
    """
    Validate a response for gibberish and repetition

    Args:
        text: Response text to validate

    Returns:
        Dict with:
        - is_gibberish: True if gibberish detected
        - is_repetitive: True if repetitive content detected
        - is_valid: True if response is valid (not gibberish and not repetitive)
    """
    is_gibberish = detect_gibberish(text)
    is_repetitive = detect_repetition(text)

    return {
        'is_gibberish': is_gibberish,
        'is_repetitive': is_repetitive,
        'is_valid': not (is_gibberish or is_repetitive)
    }


def add_validation_columns(df, response_column: str = 'response'):
    """
    Add validation columns to a DataFrame

    Args:
        df: DataFrame with responses
        response_column: Name of column containing response text

    Returns:
        DataFrame with added columns: is_gibberish, is_repetitive, is_valid
    """
    import pandas as pd

    validations = df[response_column].apply(validate_response)

    df['is_gibberish'] = validations.apply(lambda x: x['is_gibberish'])
    df['is_repetitive'] = validations.apply(lambda x: x['is_repetitive'])
    df['is_valid'] = validations.apply(lambda x: x['is_valid'])

    return df


def calculate_stratified_metrics(df, score_column: str, model_name: str = None) -> Dict:
    """
    Calculate stratified metrics for transparent reporting

    Args:
        df: DataFrame with is_valid column and score column
        score_column: Name of column containing scores
        model_name: Optional model name for display

    Returns:
        Dict with stratified metrics
    """
    # Ensure validation columns exist
    if 'is_valid' not in df.columns:
        df = add_validation_columns(df)

    total = len(df)
    valid_df = df[df['is_valid']]

    metrics = {
        'total_responses': total,
        'valid_responses': len(valid_df),
        'valid_rate': len(valid_df) / total if total > 0 else 0,
        'gibberish_rate': df['is_gibberish'].mean() if 'is_gibberish' in df.columns else 0,
        'repetitive_rate': df['is_repetitive'].mean() if 'is_repetitive' in df.columns else 0,
    }

    # Overall score (all responses)
    if score_column in df.columns:
        metrics['overall_score'] = df[score_column].mean()
        metrics['overall_std'] = df[score_column].std()

        # Valid-only score
        if len(valid_df) > 0:
            metrics['valid_score'] = valid_df[score_column].mean()
            metrics['valid_std'] = valid_df[score_column].std()
        else:
            metrics['valid_score'] = None
            metrics['valid_std'] = None

    return metrics


def print_stratified_metrics(metrics: Dict, model_name: str = "Model"):
    """
    Print stratified metrics in formatted output

    Args:
        metrics: Dict from calculate_stratified_metrics
        model_name: Model name for display
    """
    print(f"\n{model_name} Results:")
    print(f"  Total responses: {metrics['total_responses']}")
    print(f"  Valid response rate: {metrics['valid_rate']:.1%}")
    print(f"  Gibberish rate: {metrics['gibberish_rate']:.1%}")
    print(f"  Repetitive rate: {metrics['repetitive_rate']:.1%}")

    if 'overall_score' in metrics:
        print(f"  Overall score: {metrics['overall_score']:.3f} (all responses)")
        if metrics.get('valid_score') is not None:
            print(f"  Valid-only score: {metrics['valid_score']:.3f}")


def get_valid_mask(df):
    """
    Get boolean mask for valid responses

    Args:
        df: DataFrame with is_valid column (or will add it)

    Returns:
        Boolean Series mask for valid responses
    """
    if 'is_valid' not in df.columns:
        df = add_validation_columns(df)
    return df['is_valid']


def get_validation_summary(df, response_column: str = 'response') -> Dict:
    """
    Get validation summary without modifying DataFrame

    Args:
        df: DataFrame with responses
        response_column: Name of column containing response text

    Returns:
        Dict with valid_rate, gibberish_rate, repetitive_rate, n_valid, n_total
    """
    if 'is_valid' not in df.columns:
        df = add_validation_columns(df, response_column)

    total = len(df)
    n_valid = df['is_valid'].sum()

    return {
        'valid_rate': n_valid / total if total > 0 else 0,
        'gibberish_rate': df['is_gibberish'].mean() if 'is_gibberish' in df.columns else 0,
        'repetitive_rate': df['is_repetitive'].mean() if 'is_repetitive' in df.columns else 0,
        'n_valid': int(n_valid),
        'n_total': total
    }
