import spacy
import sys
import joblib
from pathlib import Path

MODEL_FILE = Path(__file__).parent/'artifacts'/'figure_model.joblib'

try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy 'en_core_web_sm' 모델 로드 성공 (예측용).")
except OSError:
    print("spaCy 모델 'en_core_web_sm'을 찾을 수 없습니다.")
    print("터미널에서 'python -m spacy download en_core_web_sm'을 실행하세요.")
    sys.exit(1)

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

def tags_to_spans(tokens, tags):
    spans = []
    current_span_tokens = []

    for i in range(len(tags)):
        tag = tags[i]
        token = tokens[i]

        if tag == 'B-REF':
            if current_span_tokens:
                spans.append(" ".join(current_span_tokens))
            current_span_tokens = [token]

        elif tag == 'I-REF':
            if current_span_tokens:
                current_span_tokens.append(token)

        else:
            if current_span_tokens:
                spans.append(" ".join(current_span_tokens))
            current_span_tokens = []

    if current_span_tokens:
        spans.append(" ".join(current_span_tokens))

    return spans


def predict_from_text(text, crf_model):
    doc = nlp(text)
    features = [token2features(doc, i) for i in range(len(doc))]
    tokens = [token.text for token in doc]
    predicted_tags = crf_model.predict([features])[0]
    spans = tags_to_spans(tokens, predicted_tags)
    return spans, tokens, predicted_tags


if __name__ == "__main__":
    try:
        crf = joblib.load(MODEL_FILE)
        print(f"모델을 '{MODEL_FILE}' 파일에서 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"--- 에러 ---")
        print(f"모델 파일 '{MODEL_FILE}'을(를) 찾을 수 없습니다.")
        print(f"`train.py`를 먼저 실행하여 모델을 생성하세요.")
        sys.exit(1)

    test_sentences = [
        "See the result in Figure 1.",
        "This is shown in the second table (see Fig. 4b).",
        "You can see from the very bottom of Figure 2.12 that the remainder is 101",
        "The first chart (Appendix A.1) shows the full data.",
        "This is not a reference, just a regular figure."
    ]

    print("\n--- 예측 결과 (로드된 모델) ---")

    for text in test_sentences:
        spans, tokens, tags = predict_from_text(text, crf)
        print(f"\n입력 텍스트: '{text}'")
        # print(f"토큰: {tokens}")  # (디버깅 시 주석 해제)
        # print(f"태그: {tags}")    # (디버깅 시 주석 해제)
        print(f"찾아낸 참조 구문: {spans}")