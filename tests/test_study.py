from rps_beliefs.study import next_participant, load_bot_lists


def test_next_participant():
    assert next_participant(0) == ("P01", 1)
    assert next_participant(1) == ("P02", 2)
    assert next_participant(9) == ("P10", 2)


def test_load_bot_lists():
    lists = load_bot_lists()
    assert lists["X"]["seed"] == 6080
    assert len(lists["Y"]["moves"]) == 150
