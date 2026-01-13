# automotive_wordcloud_analysis.py

import pandas as pd
import re
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from langdetect import detect

# ---- OPTIONAL spaCy IMPORT (SAFE) ----
try:
    import spacy
    from spacy.tokens import Token
except ImportError:
    spacy = None
    Token = None


# ---------------- CONFIGURATION ----------------

INPUT_FILE = "verbatim.xlsx"
OUTPUT_IMAGE = "automotive_wordcloud.png"
FREQ_FILE = "word_frequency_analysis.xlsx"

nlp = None
stop_words = set()


# ---------------- CUSTOM INFLECTION ----------------

def custom_inflect(token):
    """Custom inflection handler for automotive terms"""
    inflection_map = {
        'clean': 'clean',
        'wash': 'wash',
        'noise': 'noise',
        'suspension': 'suspension',
        'brake': 'brake',
        'cleaning': 'clean',
        'washing': 'wash'
    }
    return inflection_map.get(token.lemma_, token.lemma_)


# ---------------- LAZY spaCy INITIALIZER ----------------

def get_nlp():
    global nlp, stop_words

    if spacy is None:
        raise RuntimeError("spaCy is not available in local dev mode")

    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

        stop_words = nlp.Defaults.stop_words.union({
            "check", "service", "rep", "km", "vehicle", "gaadi",
            "hai", "kar", "me", "ka", "ki", "ko", "se", "ke",
            "schedule", "washing", "1000", "10000", "maxicare",
            "wheel", "alignment", "balance", "pickup", "cleaning", "wash",
            "rahi", "nhi", "rha", "krne", "rhe", "hona", "par", "lag", "clean",
            "CLU", "ENG", "BOD", "CLN", "GEN", "STG", "WHT", "IFT", "BRK",
            "ELC", "TRN", "FUE", "HVA", "SER", "EPT", "SUS", "DRL", "EXH",
            "SAF", "VAS", "RE-", "708", "013", "405", "SWU"
        })

        # Register custom token extension
        if Token is not None and not Token.has_extension("inflect"):
            Token.set_extension('inflect', default=None)

        # Configure attribute ruler
        if 'attribute_ruler' in nlp.pipe_names:
            attribute_ruler = nlp.get_pipe('attribute_ruler')
            attribute_ruler.add(
                patterns=[
                    [{'lemma': 'clean'}],
                    [{'lemma': 'wash'}],
                    [{'lemma': 'noise'}],
                    [{'lemma': 'suspension'}],
                    [{'lemma': 'brake'}],
                    [{'lemma': 'cleaning'}],
                    [{'lemma': 'washing'}]
                ],
                attrs={'_': {'inflect': custom_inflect}}
            )

    return nlp


# ---------------- DATA LOADING ----------------

def load_and_process_data():
    df = pd.read_excel(INPUT_FILE)
    df['processed_text'] = df['demanded_verbatim'].apply(
        lambda x: process_text(str(x))
    )
    return df


# ---------------- TEXT PROCESSING ----------------

def process_text(text):
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\d+', '', text).lower().strip()

    replacements = {
        r'\bcleaning\b': 'clean',
        r'\bcleans\b': 'clean',
        r'\bcleaned\b': 'clean',
        r'\bwashing\b': 'wash',
        r'\bsuspens\b': 'suspension',
        r'\bsus\b': 'suspension',
        r'\bbrk\b': 'brake',
        r'\bnoise-\b': 'noise',
        r'\bservice-\b': 'service'
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    try:
        lang = detect(text)
    except:
        lang = 'en'

    if lang == 'hi':
        return process_hindi(text)

    return process_english(text)


def process_hindi(text):
    # Placeholder for Hindi logic
    return text


def process_english(text):
    nlp_model = get_nlp()
    doc = nlp_model(text)

    processed_tokens = []

    for token in doc:
        lemma = token._.inflect if token._.inflect else token.lemma_
        lemma = lemma.lower().replace('-', '')

        if (
            len(lemma) > 2 and
            not token.is_stop and
            lemma not in stop_words
        ):
            processed_tokens.append(lemma)

    return ' '.join(processed_tokens)


# ---------------- WORDCLOUD ----------------

def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        max_words=200,
        collocations=False,
        contour_width=2,
        contour_color='#1a365f',
        colormap='viridis'
    ).generate(text)

    os.makedirs("static", exist_ok=True)
    output_path = os.path.join("static", OUTPUT_IMAGE)

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


# ---------------- FREQUENCY ANALYSIS ----------------

def analyze_frequencies(text):
    words = text.split()

    component_map = {
        'steering': ['steer', 'stg', 'steering'],
        'brake': ['brake', 'brk', 'braking'],
        'suspension': ['suspension', 'sus', 'suspens'],
        'electrical': ['electrical', 'elc', 'wiring', 'light'],
        'cooling': ['coolant', 'cooling', 'coolent'],
        'body': ['door', 'panel', 'bod', 'denting'],
        'engine': ['engine', 'pickup', 'rhs', 'turbo', 'eng'],
        'cleaning': ['clean', 'wash', 'cleaning', 'washing'],
        'noise': ['noise', 'sound', 'rattle', 'squeak']
    }

    standardized = []

    for word in words:
        matched = False
        for key, variants in component_map.items():
            if word in variants:
                standardized.append(key)
                matched = True
                break
        if not matched:
            standardized.append(word)

    return Counter(standardized)


# ---------------- MAIN ----------------

def main():
    df = load_and_process_data()
    all_text = ' '.join(df['processed_text'])

    generate_wordcloud(all_text)

    freq = analyze_frequencies(all_text)
    freq_df = pd.DataFrame(freq.most_common(), columns=['Component/Word', 'Count'])
    freq_df.to_excel(FREQ_FILE, index=False)

    print(f"Analysis complete. Results saved to {OUTPUT_IMAGE} and {FREQ_FILE}")


if __name__ == "__main__":
    main()
