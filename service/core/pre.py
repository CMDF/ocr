from service.core.ocr import *
from service.core.crop import *
from service.core.post import correct_segmentation_and_typos
from pathlib import Path

def _calculate_distance(box1, box2, flag):
    coord1 = box1['coordinate']
    coord2 = box2['coordinate']
    if flag:
        if coord1[0] > coord2[2]:
            return abs(coord1[1] - coord2[1]) + abs(coord1[3] - coord2[3])
        else:
            return abs(coord2[1] - coord1[1]) + abs(coord2[3] - coord1[3])
    if coord1[1] > coord2[3]:
        return 2*(coord1[1] - coord2[3]) + abs(coord1[0] - coord2[0]) + abs(coord1[2] - coord2[2])
    else:
        return 2*(coord2[1] - coord1[3]) + abs(coord1[0] - coord2[0]) + abs(coord1[2] - coord2[2])

def _group_adjacent_targets(boxes):
    if not boxes:
        return []

    target_labels = ['image', 'table', 'figure', 'algorithm', 'chart']
    result_boxes = []
    i = 0
    n = len(boxes)

    while i < n:
        current_box = boxes[i]
        if current_box.get('label') in target_labels:
            group_to_merge = [current_box]
            j = i+1
            while j < n and boxes[j].get('label') in target_labels:
                group_to_merge.append(boxes[j])
                j+=1

            if len(group_to_merge) == 1:
                result_boxes.append(group_to_merge[0])
            else:
                first_box = group_to_merge[0]
                min_x1 = min(box['coordinate'][0] for box in group_to_merge)
                min_y1 = min(box['coordinate'][1] for box in group_to_merge)
                max_x2 = max(box['coordinate'][2] for box in group_to_merge)
                max_y2 = max(box['coordinate'][3] for box in group_to_merge)

                merged_box = {
                    "cls_id": first_box.get('cls_id'),
                    "label": first_box.get('label'),
                    "score": first_box.get('score'),
                    "coordinate": [min_x1, min_y1, max_x2, max_y2]
                }
                result_boxes.append(merged_box)
            i=j
        else:
            result_boxes.append(current_box)
            i+=1

    return result_boxes

def group_and_sort_by_proximity(items):
    if not items:
        return []

    text = items['rec_texts']
    box = items['rec_boxes']
    coord = box[0]
    y_tolerance = (coord[3]-coord[1])*0.5
    items = list(zip(text, box))

    if not text:
        return []

    items.sort(key=lambda a: a[1][1])

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
        line.sort(key=lambda a: a[1][0])
        for box in line:
            sorted_lines.append(box)

    return sorted_lines

def group_image_with_caption(page_data: dict, folder_name: str):
    boxes = page_data.get('boxes', [])
    if not boxes:
        return page_data

    target_boxes = []
    title_boxes = []
    other_boxes = []
    for box in boxes:
        if box.get('label') in ['image', 'table', 'figure', 'algorithm', 'chart', 'formula']:
            target_boxes.append(box)
        elif box.get('label') in ['figure_title', 'figure_caption', 'table_caption', 'table_title', 'chart_caption', 'chart_title', 'formula_number']:
            title_boxes.append(box)
        else:
            other_boxes.append(box)

    merged_boxes = []
    used_title_indices = set()

    def sub_y(box1, box2):
        coord1 = box1['coordinate']
        coord2 = box2['coordinate']
        if coord1[1] > coord2[3]:
            return coord1[1] - coord2[3]
        else:
            return coord2[1] - coord1[3]

    for title_box in title_boxes:
        # if title_box['score'] < 0.8:
        #     continue
        title_coord = title_box['coordinate']
        filename = "page_" + str(page_data['page_index']+1) + ".png"
        if title_box['label'] != 'formula_number':
            figure_title_output = ocr(crop_image_by_bbox(str(Path(__file__).parent.parent.parent / "data" / "temp" / folder_name / filename), title_coord))
            figure_title_output = group_and_sort_by_proximity(figure_title_output[0])
        else:
            figure_title_output = ocr(crop_image_by_bbox(str(Path(__file__).parent.parent.parent / "data" / "temp" / folder_name / filename), title_coord))

        if not figure_title_output[0]:
            show(title_coord, str(Path(__file__).parent.parent.parent/"data"/"temp"/folder_name/filename))
            figure_title_output = [""]

        if title_box['label'] == 'formula_number':
            cal_flag = True
        else:
            cal_flag = False

        closest = min(
            ((i, target, _calculate_distance(title_box, target, cal_flag)) for i, target in enumerate(target_boxes) if i not in used_title_indices and sub_y(title_box, target) < 0.05),
            key=lambda x: x[2],
            default=(None, None, float('inf'))
        )

        if closest[1]:
            idx, target_box, _ = closest
            used_title_indices.add(idx)
            img_coord = target_box['coordinate']
            new_coord = [min(img_coord[0], title_coord[0]), min(img_coord[1], title_coord[1]),
                         max(img_coord[2], title_coord[2]), max(img_coord[3], title_coord[3])]

            figure_title = ""
            if title_box['label'] != 'formula_number':
                for res in figure_title_output:
                    figure_title = figure_title + res[0]
            else:
                for res in figure_title_output:
                    figure_title = figure_title + res['rec_texts'][0]

            def image_to_figure(a):
                if a['label'] == 'image':
                    return 'figure'
                else:
                    return a['label']

            merged_boxes.append({
                "cls_id": 99, "label": image_to_figure(target_box), "score": target_box['score'], "coordinate": new_coord,
                'text': figure_title
            })
            # show(new_coord, str(Path(__file__).parent.parent.parent/"data"/"temp"/folder_name/filename))

    unmatched_targets   = [t for i, t in enumerate(target_boxes) if i not in used_title_indices]
    for target in unmatched_targets:
        target['label'] = 'None'
    final_boxes = other_boxes + merged_boxes + unmatched_targets

    final_boxes.sort(key=lambda a: a['coordinate'][1])
    length_validation = len(final_boxes) > 3
    if length_validation:
        y_comparison_first_two = abs(final_boxes[0]['coordinate'][1] - final_boxes[1]['coordinate'][1]) < 0.0001 and final_boxes[0]['coordinate'][1] != final_boxes[1]['coordinate'][1]
        y_comparison_next_two = abs(final_boxes[1]['coordinate'][1] - final_boxes[2]['coordinate'][1]) < 0.0001 and final_boxes[1]['coordinate'][1] != final_boxes[2]['coordinate'][1]

        if not y_comparison_next_two and y_comparison_first_two:
            left_boxes = []
            right_boxes = []
            for box in final_boxes:
                if box['coordinate'][0] < 0.4:
                    left_boxes.append(box)
                else:
                    right_boxes.append(box)
            left_boxes.sort(key=lambda a: a['coordinate'][1])
            right_boxes.sort(key=lambda a: a['coordinate'][1])
            final_boxes = left_boxes + right_boxes

    result_data = page_data.copy()
    result_data['boxes'] = final_boxes

    return result_data

def _is_contained(inner_box, outer_box):
    inner_coord = inner_box['coordinate']
    outer_coord = outer_box['coordinate']

    is_x_contained = outer_coord[0]-0.0086 <= inner_coord[0] and inner_coord[2] <= outer_coord[2]+0.0086
    is_y_contained = outer_coord[1]-0.0077 <= inner_coord[1] and inner_coord[3] <= outer_coord[3]+0.0077

    return is_x_contained and is_y_contained

def remove_nested_boxes(page_data):
    boxes = page_data['boxes']
    boxes.sort(key=lambda a: a['coordinate'][1])
    if not boxes:
        return page_data
    length_validation = len(boxes) > 3
    if length_validation:
        y_comparison_first_two = abs(boxes[0]['coordinate'][1] - boxes[1]['coordinate'][1]) < 0.0001 and boxes[0]['coordinate'][1] != boxes[1]['coordinate'][1]
        y_comparison_next_two = abs(boxes[1]['coordinate'][1] - boxes[2]['coordinate'][1]) < 0.0001 and boxes[1]['coordinate'][1] != boxes[2]['coordinate'][1]
        if y_comparison_first_two and not y_comparison_next_two:
            left_boxes = []
            right_boxes = []
            for box in boxes:
                if box['coordinate'][0] < 0.4:
                    left_boxes.append(box)
                else:
                    right_boxes.append(box)
            left_boxes = _group_adjacent_targets(left_boxes)
            right_boxes = _group_adjacent_targets(right_boxes)
            boxes = left_boxes + right_boxes
    else:
        boxes = _group_adjacent_targets(boxes)

    indices_to_remove = set()

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue
            if _is_contained(boxes[i], boxes[j]):
                indices_to_remove.add(i)

    boxes_to_keep = [box for i, box in enumerate(boxes) if i not in indices_to_remove]

    final_boxes = boxes_to_keep

    result_data = page_data.copy()
    result_data['boxes'] = final_boxes

    return result_data