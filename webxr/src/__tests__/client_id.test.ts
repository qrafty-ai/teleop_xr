import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("getClientId", () => {
	const GLOBAL_KEY = "__teleop_xr_client_id__";

	beforeEach(() => {
		// Clear any cached client IDs
		const g = globalThis as unknown as Record<string, unknown>;
		delete g[GLOBAL_KEY];
		// Reset module to clear cached value
		vi.resetModules();
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("generates and caches a client ID", async () => {
		const { getClientId } = await import("../client_id");
		const id1 = getClientId();
		const id2 = getClientId();

		expect(id1).toBe(id2);
		expect(id1).toBeTruthy();
		expect(typeof id1).toBe("string");
	});

	it("uses crypto.randomUUID when available", async () => {
		const mockUUID = "550e8400-e29b-41d4-a716-446655440000";
		const originalCrypto = globalThis.crypto;

		Object.defineProperty(globalThis, "crypto", {
			value: {
				randomUUID: () => mockUUID,
			},
			writable: true,
			configurable: true,
		});

		vi.resetModules();
		const { getClientId } = await import("../client_id");

		const id = getClientId();
		expect(id).toBe(mockUUID);

		// Restore original crypto
		Object.defineProperty(globalThis, "crypto", {
			value: originalCrypto,
			writable: true,
			configurable: true,
		});
	});

	it("falls back to Math.random when crypto is not available", async () => {
		const originalCrypto = globalThis.crypto;

		// Remove crypto
		Object.defineProperty(globalThis, "crypto", {
			value: undefined,
			writable: true,
			configurable: true,
		});

		vi.resetModules();
		const { getClientId } = await import("../client_id");

		const id = getClientId();
		expect(id).toBeTruthy();
		expect(typeof id).toBe("string");
		expect(id).toMatch(/^[0-9a-f]+-[0-9a-f]+$/);

		// Restore original crypto
		Object.defineProperty(globalThis, "crypto", {
			value: originalCrypto,
			writable: true,
			configurable: true,
		});
	});

	it("reuses existing ID from globalThis", async () => {
		const existingId = "existing-client-id-12345";
		const g = globalThis as unknown as Record<string, unknown>;
		g[GLOBAL_KEY] = existingId;

		vi.resetModules();
		const { getClientId } = await import("../client_id");

		const id = getClientId();
		expect(id).toBe(existingId);
	});
});
