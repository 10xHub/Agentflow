"""Unit tests for the upstream decoupler (agentflow.core.realtime.queue)."""

import asyncio

import pytest

from agentflow.core.realtime.queue import LiveInput, LiveInputQueue


class TestLiveInputQueuePut:
    def test_send_audio_is_nonblocking_and_enqueues_audio_frame(self):
        q = LiveInputQueue()
        q.send_audio(b"\x00\x01", sample_rate=16000)
        item = q.get_nowait()
        assert isinstance(item, LiveInput)
        assert item.kind == "audio"
        assert item.data == b"\x00\x01"
        assert item.sample_rate == 16000

    def test_send_text_enqueues_text_frame(self):
        q = LiveInputQueue()
        q.send_text("hello")
        item = q.get_nowait()
        assert item.kind == "text"
        assert item.text == "hello"

    def test_send_image_enqueues_image_frame_with_mime(self):
        q = LiveInputQueue()
        q.send_image(b"\xff\xd8\xff")
        item = q.get_nowait()
        assert item.kind == "image"
        assert item.data == b"\xff\xd8\xff"
        assert item.mime_type == "image/jpeg"

    def test_send_image_accepts_custom_mime(self):
        q = LiveInputQueue()
        q.send_image(b"\x89PNG", mime_type="image/png")
        item = q.get_nowait()
        assert item.kind == "image"
        assert item.mime_type == "image/png"

    def test_activity_markers_enqueue_control_frames(self):
        q = LiveInputQueue()
        q.send_activity_start()
        q.send_activity_end()
        assert q.get_nowait().kind == "activity_start"
        assert q.get_nowait().kind == "activity_end"

    def test_close_enqueues_sentinel_and_marks_closed(self):
        q = LiveInputQueue()
        q.close()
        assert q.get_nowait().kind == "close"
        assert q.closed is True

    def test_put_after_close_is_dropped(self):
        q = LiveInputQueue()
        q.close()
        q.get_nowait()  # drain the close sentinel
        q.send_audio(b"\x00", sample_rate=16000)  # should be a no-op, not raise
        with pytest.raises(asyncio.QueueEmpty):
            q.get_nowait()


class TestLiveInputQueueConsume:
    @pytest.mark.asyncio
    async def test_get_awaits_until_item_available(self):
        q = LiveInputQueue()

        async def producer():
            await asyncio.sleep(0.01)
            q.send_text("late")

        asyncio.create_task(producer())
        item = await q.get()
        assert item.kind == "text"
        assert item.text == "late"

    @pytest.mark.asyncio
    async def test_async_iteration_stops_at_close(self):
        q = LiveInputQueue()
        q.send_text("a")
        q.send_audio(b"\x01", sample_rate=16000)
        q.close()

        seen = [item async for item in q]
        assert [i.kind for i in seen] == ["text", "audio"]
