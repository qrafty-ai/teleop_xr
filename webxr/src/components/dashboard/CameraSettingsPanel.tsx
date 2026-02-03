"use client";

import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useAppStore } from "@/lib/store";

const formatCameraLabel = (key: string) => {
	switch (key) {
		case "head":
			return "Head Camera";
		case "wrist_left":
			return "Left Wrist Camera";
		case "wrist_right":
			return "Right Wrist Camera";
		default:
			// Fallback: capitalize words, replace underscores/hyphens with spaces
			return key.replace(/[_-]/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
	}
};

export function CameraSettingsPanel() {
	const availableCameras = useAppStore((state) => state.availableCameras);
	const cameraConfig = useAppStore((state) => state.cameraConfig);
	const setCameraEnabled = useAppStore((state) => state.setCameraEnabled);
	const toggleAllCameras = useAppStore((state) => state.toggleAllCameras);

	const allEnabled =
		availableCameras.length > 0 &&
		availableCameras.every((key) => cameraConfig[key] !== false);

	return (
		<Card className="w-full">
			<CardHeader>
				<CardTitle className="flex items-center justify-between">
					<span>Camera Feeds</span>
					{availableCameras.length > 0 && (
						<Button
							variant="outline"
							size="sm"
							onClick={() => toggleAllCameras(!allEnabled)}
						>
							{allEnabled ? "Disable All" : "Enable All"}
						</Button>
					)}
				</CardTitle>
				<CardDescription>
					Toggle video streams from available cameras
				</CardDescription>
			</CardHeader>
			<CardContent className="space-y-4">
				{availableCameras.length === 0 && (
					<div className="text-sm text-muted-foreground">
						No cameras detected
					</div>
				)}
				{availableCameras.map((key) => (
					<div
						key={key}
						className="flex items-center justify-between space-x-2"
					>
						<Label htmlFor={`camera-${key}`}>{formatCameraLabel(key)}</Label>
						<Switch
							id={`camera-${key}`}
							checked={cameraConfig[key] !== false}
							onCheckedChange={(checked) => setCameraEnabled(key, checked)}
						/>
					</div>
				))}
			</CardContent>
		</Card>
	);
}
