import spacy
import joblib
from pathlib import Path
import re

MODEL_FILE = Path(__file__).parent/'artifacts'/'figure_model.joblib'
nlp = spacy.load("en_core_web_sm")
crf = joblib.load(MODEL_FILE)

def token2features(doc, i):
    token = doc[i]

    features = {
        'pos': token.pos_,
        'tag': token.tag_,
    }

    features.update({
        'bias': 1.0,
        'word.lower()': token.lower_,
        'word.isupper()': token.is_upper,
        'word.istitle()': token.is_title,
        'word.isdigit()': token.is_digit,
        'word.suffix(3)': token.text[-3:],
        'word.prefix(3)': token.text[:3],
    })

    ref_keywords = {'figure', 'fig', 'table', 'image', 'chart', 'diagram'}
    pos_keywords = {'above', 'below', 'next', 'previous', 'following', 'see', 'in'}
    ord_keywords = {'first', 'second', 'third', 'fourth', 'fifth'}

    if token.lower_ in ref_keywords:
        features['is_ref_keyword'] = True
    if token.lower_ in pos_keywords:
        features['is_pos_keyword'] = True
    if token.lower_ in ord_keywords:
        features['is_ord_keyword'] = True

    if i > 0:
        prev_token = doc[i - 1]
        features.update({
            '-1:pos': prev_token.pos_,
            '-1:tag': prev_token.tag_,
            '-1:word.lower()': prev_token.lower_,
            '-1:word.istitle()': prev_token.is_title,
            '-1:word.isdigit()': prev_token.is_digit,
            '-1:is_ref_keyword': prev_token.lower_ in ref_keywords,
        })
    else:
        features['BOS'] = True

    if i < len(doc) - 1:
        next_token = doc[i + 1]
        features.update({
            '+1:pos': next_token.pos_,
            '+1:tag': next_token.tag_,
            '+1:word.lower()': next_token.lower_,
            '+1:word.istitle()': next_token.is_title,
            '+1:word.isdigit()': next_token.is_digit,
            '+1:is_ref_keyword': next_token.lower_ in ref_keywords,
        })
    else:
        features['EOS'] = True  # 문장의 끝

    return features

class ReferenceInfo:
    def __init__(self):
        self.ref_info = []
        self.raw_texts = []
        self.section_info = []
        self.order_info = []

    def __repr__(self):
        return (f"ReferenceInfo(\n"
                f"  ref_info     = {self.ref_info},\n"
                f"  raw_text     = {self.raw_texts},\n"
                f"  section_info = {self.section_info}\n"
                f"  order_info   = {self.order_info}\n"
                f")")

def tags_to_spans(tokens, tags):
    output = ReferenceInfo()
    current_span_tokens = []

    current_span_type = None

    def save_current_span():
        if current_span_tokens and current_span_type:
            span_text = " ".join(current_span_tokens)

            if current_span_type == 'REF':
                output.ref_info.append(span_text)
            elif current_span_type == 'SEC':
                output.section_info.append(span_text)

        current_span_tokens.clear()

    for i in range(len(tags)):
        tag = tags[i]
        token = tokens[i]

        if tag == 'B-FIG' or tag == 'B-TBL':
            save_current_span()
            current_span_tokens.append(token)
            current_span_type = 'REF'

        elif tag == 'B-SEC':
            save_current_span()
            current_span_tokens.append(token)
            current_span_type = 'SEC'

        elif tag == 'I-FIG' or tag == 'I-TBL':
            if current_span_type == 'REF':
                current_span_tokens.append(token)
            else:
                save_current_span()
                current_span_type = None

        elif tag == 'I-SEC':
            if current_span_type == 'SEC':
                current_span_tokens.append(token)
            else:
                save_current_span()
                current_span_type = None

        else:
            save_current_span()
            current_span_type = None

    save_current_span()

    return output


def predict_from_text(text, crf_model=crf):
    ordinal_numbers = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
    ordinal_number_pattern = r'\b\d+(st|nd|rd|th)\b'

    doc = nlp(text)
    features = [token2features(doc, i) for i in range(len(doc))]
    tokens = [token.text for token in doc]
    predicted_tags = crf_model.predict([features])[0]
    spans = tags_to_spans(tokens, predicted_tags)
    if spans.ref_info:
        spans.raw_texts.append(text)

        for ref_info in spans.ref_info:
            for ordinal_number in ordinal_numbers:
                if ordinal_number in ref_info:
                    spans.order_info.append(ordinal_numbers.index(ordinal_number))
            match = re.search(ordinal_number_pattern, ref_info, re.IGNORECASE)
            if not spans.order_info and match:
                spans.order_info.append(int(match.group(0))-1)


    return spans, tokens, predicted_tags

if __name__ == '__main__':
    output, _, _ = predict_from_text("See the table 1.", joblib.load(MODEL_FILE))
    print(output)