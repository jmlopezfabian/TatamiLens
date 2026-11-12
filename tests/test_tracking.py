from tatami_vision.tracking import expand_to_crop, select_combat_boxes, union_box


def test_picks_ground_pair_over_referee():
    fighters = (400.0, 400.0, 800.0, 650.0)  # ancha, abajo
    referee = (900.0, 100.0, 1000.0, 450.0)  # alto, delgado, lejos
    crowd = (50.0, 50.0, 120.0, 250.0)
    chosen = select_combat_boxes([crowd, referee, fighters], 1280, 720)
    assert chosen == [fighters]


def test_includes_adjacent_second_athlete():
    a = (400.0, 400.0, 700.0, 650.0)
    b = (650.0, 380.0, 900.0, 600.0)
    assert len(select_combat_boxes([a, b], 1280, 720)) == 2
    x1, y1, x2, y2 = union_box([a, b])
    assert (x1, x2) == (400.0, 900.0)


def test_crop_clamped_and_min_size():
    x1, y1, x2, y2 = expand_to_crop((10, 10, 50, 50), 1280, 720)
    assert x1 == 0 and y1 == 0 and (x2 - x1) >= 0.4 * 1280 - 1
    assert select_combat_boxes([], 100, 100) == []
