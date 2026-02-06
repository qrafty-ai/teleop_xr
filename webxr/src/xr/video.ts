import { getClientId } from "../client_id";

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
			console.log(`[VideoClient] WebSocket connected. URL: ${url}`);
			this.waitingForOffer = false;
			this.sendControlCheck();
			this.startControlPolling();
		};
		this.ws.onerror = (err) => {
			console.error("[VideoClient] WebSocket error:", err);
		};
		this.ws.onclose = (event) => {
			console.warn(
				`[VideoClient] WebSocket closed: ${event.code} ${event.reason}`,
			);
		};
	}

	private async handleMessage(msg: { type: string; data: unknown }) {
		if (msg.type === "deny") {
			console.warn("[VideoClient] Received DENY:", msg.data);
			this.waitingForOffer = false;
			this.closePeerConnection();
			this.startControlPolling();
			return;
		}

		if (msg.type === "control_status") {
			// biome-ignore lint/suspicious/noExplicitAny: external message
			const inControl = Boolean((msg as any).data?.in_control);

			if (inControl) {
				this.stopControlPolling();
				if (!this.pc && !this.waitingForOffer) {
					console.log(
						`[VideoClient] In control, requesting video. ClientID: ${this.clientId}`,
					);
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
			console.log("[VideoClient] Received video offer");
			this.waitingForOffer = false;
			this.pc = new RTCPeerConnection();
			this.pc.ontrack = (event) => {
				console.log(
					"[VideoClient] ontrack event:",
					"kind=",
					event.track.kind,
					"id=",
					event.track.id,
					"mid=",
					event.transceiver?.mid,
					"streams=",
					event.streams.map((s) => s.id),
				);

				if (this.onTrack && event.track.kind === "video") {
					// Prioritize stream ID if available (matches python stream_ids)
					let trackId = event.track.id;
					if (event.streams.length > 0) {
						trackId = event.streams[0].id;
					} else if (event.transceiver?.mid) {
						trackId = event.transceiver.mid;
					}

					if (trackId) {
						console.log(`[VideoClient] Notifying onTrack with ID: ${trackId}`);
						this.onTrack(event.track, trackId);
					} else {
						console.warn("[VideoClient] Could not determine track ID");
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
			try {
				await this.pc.setRemoteDescription(
					msg.data as RTCSessionDescriptionInit,
				);
				const answer = await this.pc.createAnswer();
				await this.pc.setLocalDescription(answer);
				this.ws.send(
					JSON.stringify({
						type: "video_answer",
						client_id: this.clientId,
						data: this.pc.localDescription,
					}),
				);
				console.log("[VideoClient] Sent video answer");
				this.startStats();
			} catch (err) {
				console.error("[VideoClient] Error handling video offer:", err);
			}
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
