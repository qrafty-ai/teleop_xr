"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { useAppStore } from "@/lib/store";

export function AdvancedSettingsPanel() {
	const [isOpen, setIsOpen] = useState(false);

	// Access store state and actions
	const advancedSettings = useAppStore((state) => state.advancedSettings);
	const setAdvancedSettings = useAppStore((state) => state.setAdvancedSettings);

	// Settings values from store
	const { updateRate, logLevel } = advancedSettings;

	const handleUpdateRateChange = (value: number[]) => {
		setAdvancedSettings({ updateRate: value[0] });
	};

	const handleLogLevelChange = (value: number[]) => {
		const levels: ("info" | "warn" | "error")[] = ["info", "warn", "error"];
		setAdvancedSettings({ logLevel: levels[value[0]] });
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
				</CardContent>
			)}
		</Card>
	);
}
