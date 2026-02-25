"use client";

import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useAppStore } from "@/lib/store";

export function TeleopPanel() {
	const teleopSettings = useAppStore((state) => state.teleopSettings);
	const teleopLifecycle = useAppStore((state) => state.teleopLifecycle);
	const setTeleopSettings = useAppStore((state) => state.setTeleopSettings);

	const lifecycleLabel: Record<typeof teleopLifecycle, string> = {
		disconnected: "Disconnected",
		connecting: "Connecting",
		connected: "Connected",
		loading_robot: "Loading robot",
		ready: "Ready",
		reconnecting: "Reconnecting",
		error: "Connection error",
	};

	const lifecycleColorClass: Record<typeof teleopLifecycle, string> = {
		disconnected: "text-red-600",
		connecting: "text-amber-600",
		connected: "text-blue-600",
		loading_robot: "text-orange-600",
		ready: "text-green-600",
		reconnecting: "text-amber-600",
		error: "text-red-700",
	};

	const handleSpeedChange = (value: number[]) => {
		setTeleopSettings({ speed: value[0] });
	};

	const handleTurnSpeedChange = (value: number[]) => {
		setTeleopSettings({ turnSpeed: value[0] });
	};

	const handlePrecisionModeChange = (checked: boolean) => {
		setTeleopSettings({ precisionMode: checked });
	};

	return (
		<Card className="w-full">
			<CardHeader>
				<CardTitle>Teleoperation Control</CardTitle>
				<CardDescription>
					Adjust movement sensitivity and precision
				</CardDescription>
			</CardHeader>
			<CardContent className="space-y-6">
				<div className="flex items-center justify-between rounded-lg border px-3 py-2">
					<Label>Connection</Label>
					<span
						className={`text-sm font-medium ${lifecycleColorClass[teleopLifecycle]}`}
					>
						{lifecycleLabel[teleopLifecycle]}
					</span>
				</div>

				<div className="space-y-2">
					<div className="flex items-center justify-between">
						<Label htmlFor="speed-slider">Linear Speed</Label>
						<span className="text-sm text-muted-foreground">
							{Math.round(teleopSettings.speed * 100)}%
						</span>
					</div>
					<Slider
						id="speed-slider"
						min={0}
						max={2}
						step={0.01}
						value={[teleopSettings.speed]}
						onValueChange={handleSpeedChange}
					/>
				</div>

				<div className="space-y-2">
					<div className="flex items-center justify-between">
						<Label htmlFor="turn-speed-slider">Turn Speed</Label>
						<span className="text-sm text-muted-foreground">
							{Math.round(teleopSettings.turnSpeed * 100)}%
						</span>
					</div>
					<Slider
						id="turn-speed-slider"
						min={0}
						max={1}
						step={0.01}
						value={[teleopSettings.turnSpeed]}
						onValueChange={handleTurnSpeedChange}
					/>
				</div>

				<div className="flex items-center justify-between space-x-2">
					<Label htmlFor="precision-mode">Precision Mode</Label>
					<Switch
						id="precision-mode"
						checked={teleopSettings.precisionMode}
						onCheckedChange={handlePrecisionModeChange}
					/>
				</div>
			</CardContent>
		</Card>
	);
}
