export type VideoStats = {
	state: string;
	streams: Array<{
		id: string;
		fps: number;
		bitrateKbps: number;
		width: number;
		height: number;
		packetsLost: number;
		jitter: number;
	}>;
};

export class VideoClient {
	private ws: WebSocket;
	private pc: RTCPeerConnection | null = null;
	private statsTimer: number | null = null;

	constructor(
		url: string,
		private onStats: (stats: VideoStats) => void,
		private onTrack?: (track: MediaStreamTrack, trackId: string) => void,
	) {
		this.ws = new WebSocket(url);
		this.ws.onmessage = (event) => this.handleMessage(JSON.parse(event.data));
		this.ws.onopen = () => {
			this.ws.send(JSON.stringify({ type: "video_request" }));
		};
	}

	private async handleMessage(msg: { type: string; data: unknown }) {
		if (msg.type === "video_offer") {
			this.pc = new RTCPeerConnection();
			this.pc.ontrack = (event) => {
				if (this.onTrack && event.track.kind === "video") {
					// Use track id (if explicit) or transceiver mid as identifier
					const trackId = event.track.id || event.transceiver?.mid;
					if (trackId) {
						this.onTrack(event.track, trackId);
					}
				}
			};
			this.pc.onicecandidate = (e) => {
				if (e.candidate) {
					this.ws.send(
						JSON.stringify({ type: "video_ice", data: e.candidate }),
					);
				}
			};
			await this.pc.setRemoteDescription(msg.data as RTCSessionDescriptionInit);
			const answer = await this.pc.createAnswer();
			await this.pc.setLocalDescription(answer);
			this.ws.send(
				JSON.stringify({
					type: "video_answer",
					data: this.pc.localDescription,
				}),
			);
			this.startStats();
		}
		if (msg.type === "video_ice" && this.pc) {
			await this.pc.addIceCandidate(msg.data as RTCIceCandidateInit);
		}
	}

	private startStats() {
		if (!this.pc || this.statsTimer) return;
		this.statsTimer = window.setInterval(async () => {
			if (!this.pc) return;
			const report = await this.pc.getStats();
			const streams: VideoStats["streams"] = [];
			report.forEach((stat) => {
				if (stat.type === "inbound-rtp" && stat.kind === "video") {
					const bitrateKbps =
						stat.bytesReceived && stat.timestamp
							? Math.round((stat.bytesReceived * 8) / 1000)
							: 0;
					streams.push({
						id: stat.trackIdentifier || stat.ssrc?.toString() || "video",
						fps: stat.framesPerSecond || 0,
						bitrateKbps,
						width: stat.frameWidth || 0,
						height: stat.frameHeight || 0,
						packetsLost: stat.packetsLost || 0,
						jitter: stat.jitter || 0,
					});
				}
			});
			this.onStats({ state: this.pc.connectionState, streams });
		}, 1000);
	}
}
