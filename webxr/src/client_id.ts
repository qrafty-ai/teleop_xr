let cachedClientId: string | null = null;

function generateClientId(): string {
	if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
		return crypto.randomUUID();
	}

	const rand = Math.random().toString(16).slice(2);
	return `${Date.now().toString(16)}-${rand}`;
}

export function getClientId(): string {
	if (cachedClientId) {
		return cachedClientId;
	}

	const key = "__teleop_xr_client_id__";
	const g = globalThis as unknown as Record<string, unknown>;
	const existing = g[key];
	if (typeof existing === "string" && existing.length > 0) {
		cachedClientId = existing;
		return cachedClientId;
	}

	cachedClientId = generateClientId();
	g[key] = cachedClientId;
	return cachedClientId;
}
