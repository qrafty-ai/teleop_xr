import { describe, expect, it } from "vitest";

import { cn } from "../utils";

describe("cn utility", () => {
	it("merges class names correctly", () => {
		const result = cn("text-red-500", "bg-blue-500");
		expect(result).toBe("text-red-500 bg-blue-500");
	});

	it("handles conditional classes", () => {
		const result = cn("base-class", true && "conditional-class");
		expect(result).toBe("base-class conditional-class");
	});

	it("handles falsy values", () => {
		const result = cn("base-class", false && "not-included", null, undefined);
		expect(result).toBe("base-class");
	});

	it("handles tailwind merge conflicts", () => {
		const result = cn("px-2 py-1", "px-4");
		expect(result).toBe("py-1 px-4");
	});

	it("handles empty input", () => {
		const result = cn();
		expect(result).toBe("");
	});

	it("handles arrays of classes", () => {
		const result = cn(["text-sm", "font-bold"], "text-blue-500");
		expect(result).toBe("text-sm font-bold text-blue-500");
	});
});
