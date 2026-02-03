import { getClientId } from "./client_id";

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
	private clientId = getClientId();
	private inControl = false;
	private controlPollTimer: number | null = null;
	private waitingForOffer = false;

	constructor(
		url: string,
		private onStats: (stats: VideoStats) => void,
		private onTrack?: (track: MediaStreamTrack, trackId: string) => void,
	) {
		this.ws = new WebSocket(url);
		this.ws.onmessage = (event) => this.handleMessage(JSON.parse(event.data));
		this.ws.onopen = () => {
			this.inControl = false;
			this.waitingForOffer = false;
			this.sendControlCheck();
			this.startControlPolling();
		};
	}

	private async handleMessage(msg: { type: string; data: unknown }) {
		if (msg.type === "deny") {
			this.inControl = false;
			this.waitingForOffer = false;
			this.closePeerConnection();
			this.startControlPolling();
			return;
		}

		if (msg.type === "control_status") {
			// biome-ignore lint/suspicious/noExplicitAny: external message
			const inControl = Boolean((msg as any).data?.in_control);
			this.inControl = inControl;
			if (inControl) {
				this.stopControlPolling();
				if (!this.pc && !this.waitingForOffer) {
					this.waitingForOffer = true;
					this.ws.send(
						JSON.stringify({
							type: "video_request",
							client_id: this.clientId,
						}),
					);
				}
			} else {
				this.startControlPolling();
			}
			return;
		}

		if (msg.type === "video_offer") {
			this.waitingForOffer = false;
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
						JSON.stringify({
							type: "video_ice",
							client_id: this.clientId,
							data: e.candidate,
						}),
					);
				}
			};
			await this.pc.setRemoteDescription(msg.data as RTCSessionDescriptionInit);
			const answer = await this.pc.createAnswer();
			await this.pc.setLocalDescription(answer);
			this.ws.send(
				JSON.stringify({
					type: "video_answer",
					client_id: this.clientId,
					data: this.pc.localDescription,
				}),
			);
			this.startStats();
		}
		if (msg.type === "video_ice" && this.pc) {
			await this.pc.addIceCandidate(msg.data as RTCIceCandidateInit);
		}
	}

	private startControlPolling() {
		if (this.controlPollTimer !== null) {
			return;
		}
		this.controlPollTimer = window.setInterval(() => {
			this.sendControlCheck();
		}, 1000);
	}

	private stopControlPolling() {
		if (this.controlPollTimer === null) {
			return;
		}
		window.clearInterval(this.controlPollTimer);
		this.controlPollTimer = null;
	}

	private sendControlCheck() {
		if (this.ws.readyState !== WebSocket.OPEN) {
			return;
		}
		this.ws.send(
			JSON.stringify({
				type: "control_check",
				client_id: this.clientId,
				data: {},
			}),
		);
	}

	private closePeerConnection() {
		if (this.statsTimer !== null) {
			window.clearInterval(this.statsTimer);
			this.statsTimer = null;
		}
		if (this.pc) {
			this.pc.close();
			this.pc = null;
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
