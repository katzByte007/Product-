import time

from backend.stream import StreamPublisher


def test_idle_publisher_uses_lower_encode_rate():
    publisher = StreamPublisher("cam-1")

    assert publisher._publish_interval(time.monotonic()) == publisher._idle_interval
    assert publisher._idle_interval >= 1.0


def test_recent_jpeg_request_restores_configured_rate():
    publisher = StreamPublisher("cam-1")
    requested_at = time.monotonic()

    publisher.get_jpeg()

    assert publisher._publish_interval(requested_at + 0.1) == publisher._interval
    assert publisher._wake_event.is_set()