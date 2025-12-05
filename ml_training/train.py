import sklearn_crfsuite, spacy, sys, joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
from spacy.tokens import Doc

CONLL_FILE = 'train_data.conll'
MODEL_FILE = Path(__file__).parent.parent/"service"/"models"/"artifacts"/"figure_model.joblib"

try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy 'en_core_web_sm' 모델 로드 성공.")
except OSError:
    print("spaCy 모델 'en_core_web_sm'을 찾을 수 없습니다.")
    print("터미널에서 'python -m spacy download en_core_web_sm'을 실행하세요.")
    sys.exit(1)


def load_data(filepath):
    sentences = []
    current_sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    token = parts[0]
                    label = parts[3]
                    current_sentence.append((token, label))
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []

        if current_sentence:
            sentences.append(current_sentence)

    return sentences


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
        features['EOS'] = True

    return features


def sent2features(s):
    tokens = [token for token, label in s]
    doc = Doc(nlp.vocab, words=tokens)
    for i in doc:
        print(i)
    # raw_text = " ".join(tokens)
    # doc = nlp(raw_text)

    if len(doc) != len(tokens):
        print("--- 🚨 토큰화 불일치 경고 (학습 데이터)! ---")
        print(f".conll 토큰 ({len(tokens)}개): {tokens}")
        print(f"spaCy 토큰 ({len(doc)}개): {[t.text for t in doc]}")
        return None

    return [token2features(doc, i) for i in range(len(doc))]


def sent2labels(s):
    return [label for token, label in s]


if __name__ == "__main__":
    print(f"'{CONLL_FILE}'에서 학습 데이터를 로드합니다...")
    sentences = load_data(CONLL_FILE)
    X = []
    y = []
    for s in sentences:
        features = sent2features(s)
        if features:
            X.append(features)
            y.append(sent2labels(s))

    print(f"총 {len(sentences)}개 문장 중 {len(X)}개 문장 로드 완료.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"학습 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")

    print("\nCRF 모델 학습을 시작합니다...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    print("모델 학습 완료!")

    joblib.dump(crf, MODEL_FILE)
    print(f"\n모델을 '{MODEL_FILE}' 파일로 성공적으로 저장했습니다.")

    y_pred = crf.predict(X_test)
    print("\n--- 모델 성능 평가 (Test Set) ---")

    labels = [label for label in crf.classes_ if label in ['O', 'B-FIG', 'I-FIG']]
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

    report = sklearn_crfsuite.metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    )
    print(report)