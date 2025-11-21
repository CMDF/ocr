from symspellpy import SymSpell
from pathlib import Path
import re
import numpy as np

sym_spell = SymSpell(max_dictionary_edit_distance=2,
                     prefix_length=7)

dictionary_path = Path(__file__).parent.parent.parent/"data"/"en-80k.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_segmentation_and_typos(raw_text: str, sym_spell_instance=sym_spell):
    if not raw_text:
        return ""

    pattern = r'([a-zA-Z]+|[^a-zA-Z]+)'
    tokens = re.findall(pattern, raw_text)

    correct_tokens = []
    for token in tokens:
        if token.isalpha():
            correct_token = sym_spell_instance.word_segmentation(token).corrected_string
        else:
            correct_token = token
        correct_tokens.append(correct_token)

    correct_tokens = [token.strip() for token in correct_tokens]
    correct_tokens = [token for token in correct_tokens if token]

    return ' '.join(correct_tokens)

def correct(target, line_y_tolerance_ratio=0.3, space_threshold_ratio=0.35):
    rec_texts = target["rec_texts"]
    rec_boxes = target["rec_boxes"]

    items = list(zip(rec_boxes, rec_texts))
    items.sort(key=lambda a: a[0][1])

    lines = []
    if [box[3] - box[1] for box, text in items]:
        avg_line_height = np.mean([box[3] - box[1] for box, text in items])
    else:
        raise Exception
    line_y_tolerance = avg_line_height * line_y_tolerance_ratio

    try:
        current_line_y_center = (items[0][0][1] + items[0][0][3]) / 2.0
    except Exception:
        raise
    current_line = [items[0]]

    for item in items[1:]:
        box, text = item
        box_y_center = (box[1] + box[3]) / 2.0

        if abs(box_y_center - current_line_y_center) <= line_y_tolerance:
            current_line.append(item)
            current_line_y_center = (current_line_y_center * (len(current_line) - 1) + box_y_center) / len(current_line)
        else:
            lines.append(current_line)
            current_line = [item]
            current_line_y_center = box_y_center

    lines.append(current_line)

    corrected_lines = []
    for line in lines:
        line.sort(key=lambda a: a[0][0])

        reconstructed_line_text = ""
        previous_box_x_max = -1

        for item in line:
            box, text = item
            current_box_x_min = box[0]
            current_box_x_max = box[2]
            box_height = box[3] - box[1]

            space_threshold = box_height * space_threshold_ratio

            if previous_box_x_max == -1:
                reconstructed_line_text = text
            else:
                horizontal_gap = current_box_x_min - previous_box_x_max

                if horizontal_gap > space_threshold:
                    reconstructed_line_text += " " + text
                else:
                    reconstructed_line_text += text

            previous_box_x_max = current_box_x_max

        final = correct_segmentation_and_typos(str(reconstructed_line_text), sym_spell)
        corrected_lines.append(final)

    return corrected_lines