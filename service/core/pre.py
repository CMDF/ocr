from service.core.ocr import *
from service.core.crop import *
from service.core.post import correct_segmentation_and_typos
from pathlib import Path

def _calculate_distance(box1, box2):
    coord1 = box1['coordinate']
    coord2 = box2['coordinate']
    center1_x = (coord1[0] + coord1[2]) / 2
    center1_y = (coord1[1] + coord1[3]) / 2
    center2_x = (coord2[0] + coord2[2]) / 2
    center2_y = (coord2[1] + coord2[3]) / 2
    distance = math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
    return distance

def group_and_sort_by_proximity(items, y_tolerance=5):
    if not items:
        return []

    text = items['rec_texts']
    box = items['rec_boxes']
    items = list(zip(text, box))

    items.sort(key=lambda item: item[1][1])

    lines = []

    try:
        current_line_y_ref = items[0][1][1]
    except Exception as e:
        raise e
    current_line = [items[0]]

    for item in items[1:]:
        y_value = item[1][1]

        if abs(y_value - current_line_y_ref) <= y_tolerance:
            current_line.append(item)
            current_line_y_ref = sum(i[1][1] for i in current_line) / len(current_line)
        else:
            lines.append(current_line)
            current_line = [item]
            current_line_y_ref = item[1][1]

    if current_line:
        lines.append(current_line)

    sorted_lines = []
    for line in lines:
        line.sort(key=lambda item: item[1][0])
        for box in line:
            sorted_lines.append(box)

    return sorted_lines

def group_image_with_caption(page_data):
    boxes = page_data.get('boxes', [])
    if not boxes:
        return page_data

    image_boxes = [b for b in boxes if b.get('label') == 'image']
    title_boxes = [b for b in boxes if b.get('label') == 'figure_title']
    other_boxes = [b for b in boxes if b.get('label') not in ['image', 'figure_title']]

    merged_boxes = []
    used_title_indices = set()

    for image_box in image_boxes:
        closest = min(
            ((i, title, _calculate_distance(image_box, title)) for i, title in enumerate(title_boxes) if
             i not in used_title_indices),
            key=lambda x: x[2],
            default=(None, None, float('inf'))
        )

        if closest[1]:
            idx, title_box, _ = closest
            used_title_indices.add(idx)
            img_coord, title_coord = image_box['coordinate'], title_box['coordinate']
            new_coord = [min(img_coord[0], title_coord[0]), min(img_coord[1], title_coord[1]),
                         max(img_coord[2], title_coord[2]), max(img_coord[3], title_coord[3])]
            filename = "page_" + str(page_data['page_index'] + 1) + ".png"
            path = Path(__file__).parent.parent.parent/"data"/"temp"/"Test"/filename

            figure_area = crop_image_by_bbox(str(path), title_coord)
            figure_title_output = ocr(figure_area)
            try:
                figure_title_output = group_and_sort_by_proximity(figure_title_output[0])
            except Exception as e:
                show(path=str(path), coordinate=title_coord)
                show(path=str(path), coordinate=img_coord)
                show(path=str(path), coordinate=new_coord)
                print(e)
                exit(1)

            figure_title = ""
            for res in figure_title_output:
                figure_title = figure_title + res[0]
            # print(correct_segmentation_and_typos(figure_title))

            merged_boxes.append({
                "cls_id": 99, "label": "figure", "score": image_box['score'], "coordinate": new_coord, 'text': correct_segmentation_and_typos(figure_title)
            })

    unmatched_titles = [t for i, t in enumerate(title_boxes) if i not in used_title_indices]
    final_boxes = other_boxes + merged_boxes + unmatched_titles

    final_boxes.sort(key=lambda box: box['coordinate'][1])

    result_data = page_data.copy()
    result_data['boxes'] = final_boxes
    return result_data

def _is_contained(inner_box, outer_box):
    inner_coord = inner_box['coordinate']
    outer_coord = outer_box['coordinate']

    is_x_contained = outer_coord[0] <= inner_coord[0] and inner_coord[2] <= outer_coord[2]
    is_y_contained = outer_coord[1] <= inner_coord[1] and inner_coord[3] <= outer_coord[3]

    return is_x_contained and is_y_contained

def remove_nested_boxes(page_data):
    boxes = page_data['boxes']

    text_boxes = [box for box in boxes if box['label'] == 'text']
    other_boxes = [box for box in boxes if box['label'] != 'text']

    boxes_to_keep = []

    for other_box in other_boxes:
        is_nested = False
        for text_box in text_boxes:
            if _is_contained(other_box, text_box):
                is_nested = True
                break

        if not is_nested:
            boxes_to_keep.append(other_box)

    final_boxes = text_boxes + boxes_to_keep
    final_boxes.sort(key=lambda box: box['coordinate'][1])

    result_data = page_data.copy()
    result_data['boxes'] = final_boxes

    return result_data