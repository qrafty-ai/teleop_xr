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

export function AdvancedSettingsPanel() {
	const [isOpen, setIsOpen] = useState(false);

	// Access store state and actions
	const advancedSettings = useAppStore((state) => state.advancedSettings);
	const setAdvancedSettings = useAppStore((state) => state.setAdvancedSettings);
	const setRobotResetTrigger = useAppStore(
		(state) => state.setRobotResetTrigger,
	);

	// Settings values from store
	const { updateRate, logLevel, robotVisible, spawnDistance, spawnHeight } =
		advancedSettings;

	const handleUpdateRateChange = (value: number[]) => {
		setAdvancedSettings({ updateRate: value[0] });
	};

	const handleLogLevelChange = (value: number[]) => {
		const levels: ("info" | "warn" | "error")[] = ["info", "warn", "error"];
		setAdvancedSettings({ logLevel: levels[value[0]] });
	};

	const handleRobotVisibilityChange = (checked: boolean) => {
		setAdvancedSettings({ robotVisible: checked });
	};

	const handleSpawnDistanceChange = (value: number[]) => {
		setAdvancedSettings({ spawnDistance: value[0] });
	};

	const handleSpawnHeightChange = (value: number[]) => {
		setAdvancedSettings({ spawnHeight: value[0] });
	};

	const handleResetRobot = () => {
		setRobotResetTrigger(Date.now());
	};

	const getLogLevelIndex = (level: string) => {
		const levels = ["info", "warn", "error"];
		return Math.max(0, levels.indexOf(level));
	};

	return (
		<Card className="w-full">
			<CardHeader
				className="cursor-pointer select-none"
				onClick={() => setIsOpen(!isOpen)}
			>
				<div className="flex items-center justify-between">
					<div>
						<CardTitle>Advanced Settings</CardTitle>
						<CardDescription>Network and visualization tuning</CardDescription>
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
					{/* Network Section */}
					<div className="space-y-4">
						<h4 className="text-sm font-medium leading-none text-muted-foreground">
							Network
						</h4>

						<div className="space-y-2">
							<div className="flex items-center justify-between">
								<Label htmlFor="update-rate">Update Rate</Label>
								<span className="text-sm text-muted-foreground">
									{updateRate} Hz
								</span>
							</div>
							<Slider
								id="update-rate"
								min={30}
								max={100}
								step={1}
								value={[updateRate]}
								onValueChange={handleUpdateRateChange}
							/>
						</div>

						<div className="space-y-2">
							<div className="flex items-center justify-between">
								<Label htmlFor="log-level">Log Level</Label>
								<span className="text-sm text-muted-foreground">
									{logLevel.toUpperCase()}
								</span>
							</div>
							<Slider
								id="log-level"
								min={0}
								max={2}
								step={1}
								value={[getLogLevelIndex(logLevel)]}
								onValueChange={handleLogLevelChange}
							/>
							<div className="flex justify-between px-1 text-xs text-muted-foreground">
								<span>Info</span>
								<span>Warn</span>
								<span>Error</span>
							</div>
						</div>
					</div>

					{/* Visualization Section */}
					<div className="space-y-4">
						<h4 className="text-sm font-medium leading-none text-muted-foreground">
							Visualization
						</h4>

						<div className="flex items-center justify-between space-x-2">
							<Label htmlFor="robot-visible">Robot Visibility</Label>
							<Switch
								id="robot-visible"
								checked={robotVisible}
								onCheckedChange={handleRobotVisibilityChange}
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
