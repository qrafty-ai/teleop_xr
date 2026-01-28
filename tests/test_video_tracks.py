from teleop.video_stream import VideoStreamConfig, build_tracks


def test_build_tracks_skips_disabled():
    configs = [
        VideoStreamConfig(id="cam0", device=0, enabled=True),
        VideoStreamConfig(id="cam1", device=1, enabled=False),
    ]
    tracks = build_tracks(configs)
    assert [t.stream_id for t in tracks] == ["cam0"]
