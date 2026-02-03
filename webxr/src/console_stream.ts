import { getClientId } from "./client_id";

/**
 * Console log streaming utility for Quest VR debugging.
 * Intercepts console.log/warn/error and sends to Python server via WebSocket.
 */

let ws: WebSocket | null = null;
let messageQueue: string[] = [];
const clientId = getClientId();
const originalConsole = {
	log: console.log.bind(console),
	warn: console.warn.bind(console),
	error: console.error.bind(console),
	info: console.info.bind(console),
};

// biome-ignore lint/suspicious/noExplicitAny: Console args are any
function sendLog(level: string, args: any[]) {
	const message = args
		.map((arg) => (typeof arg === "object" ? JSON.stringify(arg) : String(arg)))
		.join(" ");

	const payload = JSON.stringify({
		type: "console_log",
		client_id: clientId,
		data: { level, message },
	});

	if (ws && ws.readyState === WebSocket.OPEN) {
		ws.send(payload);
	} else {
		// Queue messages until connected
		messageQueue.push(payload);
	}
}

export function initConsoleStream() {
	const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
	const wsUrl = `${protocol}//${window.location.host}/ws`;

	ws = new WebSocket(wsUrl);

	ws.onopen = () => {
		// Flush queued messages
		for (const msg of messageQueue) {
			ws?.send(msg);
		}
		messageQueue = [];
		originalConsole.log("[ConsoleStream] Connected");
	};

	ws.onclose = () => {
		originalConsole.warn("[ConsoleStream] Disconnected, reconnecting...");
		setTimeout(initConsoleStream, 3000);
	};

	ws.onerror = (e) => {
		originalConsole.error("[ConsoleStream] Error", e);
	};

	// Intercept console methods
	// biome-ignore lint/suspicious/noExplicitAny: Console methods accept any arguments
	console.log = (...args: any[]) => {
		originalConsole.log(...args);
		sendLog("log", args);
	};

	// biome-ignore lint/suspicious/noExplicitAny: Console methods accept any arguments
	console.warn = (...args: any[]) => {
		originalConsole.warn(...args);
		sendLog("warn", args);
	};

	// biome-ignore lint/suspicious/noExplicitAny: Console methods accept any arguments
	console.error = (...args: any[]) => {
		originalConsole.error(...args);
		sendLog("error", args);
	};

	// biome-ignore lint/suspicious/noExplicitAny: Console methods accept any arguments
	console.info = (...args: any[]) => {
		originalConsole.info(...args);
		sendLog("info", args);
	};
}
