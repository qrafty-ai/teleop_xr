"use client";

import { ChevronDown, ChevronUp, RefreshCw } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
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

export function RobotSettingsPanel() {
	const [isOpen, setIsOpen] = useState(false);

	// Access store state and actions
	const robotSettings = useAppStore((state) => state.robotSettings);
	const setRobotSettings = useAppStore((state) => state.setRobotSettings);
	const setRobotResetTrigger = useAppStore(
		(state) => state.setRobotResetTrigger,
	);

	// Settings values from store
	const {
		robotVisible,
		showAxes,
		spawnDistance,
		spawnHeight,
		distanceGrabEnabled,
	} = robotSettings;

	const handleRobotVisibilityChange = (checked: boolean) => {
		setRobotSettings({ robotVisible: checked });
	};

	const handleShowAxesChange = (checked: boolean) => {
		setRobotSettings({ showAxes: checked });
	};

	const handleDistanceGrabChange = (checked: boolean) => {
		setRobotSettings({ distanceGrabEnabled: checked });
	};

	const handleSpawnDistanceChange = (value: number[]) => {
		setRobotSettings({ spawnDistance: value[0] });
	};

	const handleSpawnHeightChange = (value: number[]) => {
		setRobotSettings({ spawnHeight: value[0] });
	};

	const handleResetRobot = () => {
		setRobotResetTrigger(Date.now());
	};

	return (
		<Card className="w-full">
			<CardHeader
				className="cursor-pointer select-none"
				onClick={() => setIsOpen(!isOpen)}
			>
				<div className="flex items-center justify-between">
					<div>
						<CardTitle>Robot Settings</CardTitle>
						<CardDescription>Visibility and positioning</CardDescription>
					</div>
					{isOpen ? (
						<ChevronUp className="h-4 w-4" />
					) : (
						<ChevronDown className="h-4 w-4" />
					)}
				</div>
			</CardHeader>
			{isOpen && (
				<CardContent className="space-y-6">
					{/* Visualization Section */}
					<div className="space-y-4">
						<div className="flex items-center justify-between space-x-2">
							<Label htmlFor="robot-visible">Robot Visibility</Label>
							<Switch
								id="robot-visible"
								checked={robotVisible}
								onCheckedChange={handleRobotVisibilityChange}
							/>
						</div>

						<div className="flex items-center justify-between space-x-2">
							<Label htmlFor="show-axes">Show Axes</Label>
							<Switch
								id="show-axes"
								checked={showAxes}
								onCheckedChange={handleShowAxesChange}
							/>
						</div>

						<div className="flex items-center justify-between space-x-2">
							<Label htmlFor="distance-grab">Distance Grab</Label>
							<Switch
								id="distance-grab"
								checked={distanceGrabEnabled}
								onCheckedChange={handleDistanceGrabChange}
							/>
						</div>

						<div className="space-y-2">
							<div className="flex items-center justify-between">
								<Label htmlFor="spawn-distance">Spawn Distance</Label>
								<span className="text-sm text-muted-foreground">
									{spawnDistance.toFixed(1)}m
								</span>
							</div>
							<Slider
								id="spawn-distance"
								min={0.5}
								max={3.0}
								step={0.1}
								value={[spawnDistance]}
								onValueChange={handleSpawnDistanceChange}
							/>
						</div>

						<div className="space-y-2">
							<div className="flex items-center justify-between">
								<Label htmlFor="spawn-height">Spawn Height Offset</Label>
								<span className="text-sm text-muted-foreground">
									{spawnHeight.toFixed(1)}m
								</span>
							</div>
							<Slider
								id="spawn-height"
								min={-1.0}
								max={1.0}
								step={0.1}
								value={[spawnHeight]}
								onValueChange={handleSpawnHeightChange}
							/>
						</div>
					</div>

					{/* Actions */}
					<div className="pt-2">
						<Button
							variant="outline"
							className="w-full gap-2"
							onClick={handleResetRobot}
						>
							<RefreshCw className="h-4 w-4" />
							Reset Robot Position
						</Button>
					</div>
				</CardContent>
			)}
		</Card>
	);
}
