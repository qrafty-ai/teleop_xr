"use client";

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

const CAMERA_KEYS = [
	{ key: "head", label: "Head Camera" },
	{ key: "wrist_left", label: "Left Wrist Camera" },
	{ key: "wrist_right", label: "Right Wrist Camera" },
];

export function CameraSettingsPanel() {
	const getCameraEnabled = useAppStore((state) => state.getCameraEnabled);
	const setCameraEnabled = useAppStore((state) => state.setCameraEnabled);

	return (
		<Card className="w-full">
			<CardHeader>
				<CardTitle>Camera Feeds</CardTitle>
				<CardDescription>
					Toggle video streams from available cameras
				</CardDescription>
			</CardHeader>
			<CardContent className="space-y-4">
				{CAMERA_KEYS.map(({ key, label }) => (
					<div
						key={key}
						className="flex items-center justify-between space-x-2"
					>
						<Label htmlFor={`camera-${key}`}>{label}</Label>
						<Switch
							id={`camera-${key}`}
							checked={getCameraEnabled(key)}
							onCheckedChange={(checked) => setCameraEnabled(key, checked)}
						/>
					</div>
				))}
			</CardContent>
		</Card>
	);
}
