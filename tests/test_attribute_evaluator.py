from evaluate_attribute_models import box_iou, score_image


def test_box_iou_handles_overlapping_and_disjoint_boxes():
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175
    assert box_iou((0, 0, 2, 2), (3, 3, 4, 4)) == 0


def test_score_image_greedily_matches_each_prompt_class():
    annotations = [
        {"label": "white helmet", "box": (0, 0, 10, 10)},
        {"label": "blue gloves", "box": (20, 20, 30, 30)},
    ]
    detections = [
        {"label": "white helmet", "box": (0, 0, 10, 10), "score": 0.9},
        {"label": "white helmet", "box": (40, 40, 50, 50), "score": 0.7},
    ]

    result = score_image(annotations, detections, 0.5, ["white helmet", "blue gloves"])

    assert result["white helmet"] == {"tp": 1, "fp": 1, "fn": 0}
    assert result["blue gloves"] == {"tp": 0, "fp": 0, "fn": 1}