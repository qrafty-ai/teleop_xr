import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		environment: "node",
		include: ["src/**/*.test.ts"],
		coverage: {
			provider: "v8",
			reporter: ["text", "json", "html", "lcov"],
			include: ["src/**/*.ts"],
			exclude: [
				"src/**/*.test.ts",
				"src/**/*.config.ts",
				"**/__tests__/**",
				"**/node_modules/**",
				// Exclude complex integration/system files that heavily depend on external libraries
				"src/xr/index.ts", // Main integration file
				"src/xr/panel.ts", // ECS system with Three.js
				"src/xr/panels.ts", // Complex UI panels with Three.js/ECS
				"src/xr/robot_system.ts", // Complex ECS robot system
				"src/xr/teleop_system.ts", // Complex ECS teleop system
				"src/xr/controller_camera_system.ts", // Complex ECS controller system
				"src/xr/camera_settings_system.ts", // Small ECS system
				"src/xr/video.ts", // Complex WebRTC video client
				"src/xr/console_stream.ts", // Console streaming infrastructure
			],
			thresholds: {
				lines: 80,
				functions: 80,
				branches: 80,
				statements: 80,
			},
		},
	},
});
