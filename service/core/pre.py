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

def _group_adjacent_targets(boxes, distance_thresh):
    if not boxes:
        return []

    merged_boxes = list(boxes)

    while True:
        was_merged_in_pass = False
        i = 0
        while i < len(merged_boxes):
            j = i+1
            while j < len(merged_boxes):
                if _calculate_distance(merged_boxes[i], merged_boxes[i+1]) < distance_thresh:
                    new_coord = [
                        min(merged_boxes[i]['coordinate'][0], merged_boxes[i+1]['coordinate'][0]),
                        min(merged_boxes[i]['coordinate'][1], merged_boxes[i+1]['coordinate'][1]),
                        max(merged_boxes[i]['coordinate'][2], merged_boxes[i+1]['coordinate'][2]),
                        max(merged_boxes[i]['coordinate'][3], merged_boxes[i+1]['coordinate'][3])
                    ]

                    merged_boxes[i]['coordinate'] = new_coord
                    del merged_boxes[j]
                    was_merged_in_pass = True
                    break
                else:
                    j+=1
            if was_merged_in_pass:
                break
            else:
                i+=1

        if not was_merged_in_pass:
            break

    return merged_boxes

def group_and_sort_by_proximity(items, y_tolerance=5):
    if not items:
        return []

    text = items['rec_texts']
    box = items['rec_boxes']
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

    target_boxes    = [b for b in boxes if b.get('label') in ['image', 'table', 'figure', 'algorithm', 'chart']]
    title_boxes     = [b for b in boxes if b.get('label') in ['figure_title', 'figure_caption', 'table_caption',
                                                              'table_title', 'chart_caption', 'chart_title']]
    other_boxes     = [b for b in boxes if b.get('label') not in ['image', 'table', 'figure', 'algorithm', 'chart',
                                                                  'figure_title', 'figure_caption', 'table_caption',
                                                                  'table_title', 'chart_caption', 'chart_title']]

    # table_boxes = _group_adjacent_targets(table_boxes, 0.5)
    # target_boxes = _group_adjacent_targets(target_boxes, 0.5)

    merged_boxes = []
    used_title_indices = set()

    for title_box in title_boxes:
        title_coord = title_box['coordinate']
        filename = "page_" + str(page_data['page_index']+1) + ".png"
        figure_title_output = ocr(crop_image_by_bbox(str(Path(__file__).parent.parent.parent/"data"/"temp"/folder_name/filename), title_coord))
        figure_title_output = group_and_sort_by_proximity(figure_title_output[0])
        if not figure_title_output:
            continue

        closest = min(
            ((i, target, _calculate_distance(title_box, target)) for i, target in enumerate(target_boxes) if
             i not in used_title_indices),
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
            for res in figure_title_output:
                figure_title = figure_title + res[0]

            def image_to_figure(a):
                if a['label'] == 'image':
                    return 'figure'
                else:
                    return a['label']

            merged_boxes.append({
                "cls_id": 99, "label": image_to_figure(target_box), "score": target_box['score'], "coordinate": new_coord,
                'text': correct_segmentation_and_typos(figure_title)
            })

    unmatched_targets   = [t for i, t in enumerate(target_boxes) if i not in used_title_indices]
    for target in unmatched_targets:
        target['label'] = 'None'
    final_boxes = other_boxes + merged_boxes + unmatched_targets

    final_boxes.sort(key=lambda box: box['coordinate'][1])

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
    # Todo: 여기서 가까운 figure들 합치고 진행
    boxes = page_data['boxes']

    indices_to_remove = set()

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue

            if _is_contained(boxes[i], boxes[j]):
                box_i_coord = boxes[i]['coordinate']
                box_j_coord = boxes[j]['coordinate']

                area_i = (box_i_coord[2] - box_i_coord[0]) * (box_i_coord[3] - box_i_coord[1])
                area_j = (box_j_coord[2] - box_j_coord[0]) * (box_j_coord[3] - box_j_coord[1])
                
                if area_i < area_j:
                    indices_to_remove.add(i)
                elif area_i == area_j and i > j:
                    indices_to_remove.add(i)

    boxes_to_keep = [box for i, box in enumerate(boxes) if i not in indices_to_remove]

    final_boxes = boxes_to_keep
    final_boxes.sort(key=lambda box: box['coordinate'][1])

    result_data = page_data.copy()
    result_data['boxes'] = final_boxes

    return result_data