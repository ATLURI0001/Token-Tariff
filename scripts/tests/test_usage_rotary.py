from scripts.usage_rotary import UsageRotary


def test_usage_rotary():
    r = UsageRotary(capacity=3)
    for i in range(5):
        r.record(i * 100)
    assert len(r) == 3
