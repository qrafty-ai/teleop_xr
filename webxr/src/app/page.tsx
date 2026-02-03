"use client";

import { Rocket } from "lucide-react";
import dynamic from "next/dynamic";
import { useState } from "react";
import { CameraSettingsPanel } from "@/components/dashboard/CameraSettingsPanel";
import { TeleopPanel } from "@/components/dashboard/TeleopPanel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useAppStore } from "@/lib/store";

const XRScene = dynamic(
	() => import("@/components/xr/XRScene").then((mod) => mod.XRScene),
	{ ssr: false },
);

export default function Home() {
	const [isImmersiveRequested, setIsImmersiveRequested] = useState(false);
	const isPassthroughEnabled = useAppStore(
		(state) => state.isPassthroughEnabled,
	);
	const isImmersiveActive = useAppStore((state) => state.isImmersiveActive);
	const setIsPassthroughEnabled = useAppStore(
		(state) => state.setIsPassthroughEnabled,
	);

	return (
		<main
			className={`min-h-screen p-8 ${
				isImmersiveActive ? "bg-transparent" : "bg-background"
			}`}
		>
			<XRScene
				onExitAction={() => setIsImmersiveRequested(false)}
				isImmersiveRequested={isImmersiveRequested}
				passthrough={isPassthroughEnabled}
			/>
			<div className="relative z-10 mx-auto max-w-5xl space-y-8">
				<header className="flex items-center justify-between border-b pb-6">
					<div>
						<h1 className="text-3xl font-bold tracking-tight">TeleopXR</h1>
						<p className="text-muted-foreground">
							Robot Teleoperation Dashboard
						</p>
					</div>
					<div className="flex items-center gap-4">
						<div className="flex items-center space-x-2">
							<Switch
								id="passthrough-mode"
								checked={isPassthroughEnabled}
								onCheckedChange={setIsPassthroughEnabled}
							/>
							<Label htmlFor="passthrough-mode">Passthrough</Label>
						</div>
						<Button
							size="lg"
							className="gap-2"
							onClick={() => setIsImmersiveRequested(!isImmersiveRequested)}
							variant={isImmersiveRequested ? "destructive" : "default"}
						>
							<Rocket className="h-4 w-4" />
							{isImmersiveRequested ? "Exit XR" : "Enter XR"}
						</Button>
					</div>
				</header>

				<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
					<div className="space-y-6 lg:col-span-2">
						<TeleopPanel />
					</div>
					<div className="space-y-6">
						<CameraSettingsPanel />
					</div>
				</div>
			</div>
		</main>
	);
}
