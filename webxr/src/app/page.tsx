"use client";

import { Glasses, Monitor, X } from "lucide-react";
import dynamic from "next/dynamic";
import { useCallback, useState } from "react";
import { AdvancedSettingsPanel } from "@/components/dashboard/AdvancedSettingsPanel";
import { CameraSettingsPanel } from "@/components/dashboard/CameraSettingsPanel";
import { TeleopPanel } from "@/components/dashboard/TeleopPanel";
import { Button } from "@/components/ui/button";

const XRScene = dynamic(
	() => import("@/components/xr/XRScene").then((mod) => mod.XRScene),
	{ ssr: false },
);

export type XRMode = "vr" | "passthrough" | null;

export default function Home() {
	const [xrError, setXrError] = useState<string | null>(null);
	const [selectedMode, setSelectedMode] = useState<XRMode>(null);

	const handleEnterVR = useCallback(() => {
		setXrError(null);
		setSelectedMode("vr");
	}, []);

	const handleEnterPassthrough = useCallback(() => {
		setXrError(null);
		setSelectedMode("passthrough");
	}, []);

	const handleExit = useCallback(() => {
		setSelectedMode(null);
	}, []);

	return (
		<main className="min-h-screen bg-transparent p-8">
			{selectedMode && (
				<XRScene
					mode={selectedMode}
					onError={(message) => setXrError(message)}
					onExit={handleExit}
				/>
			)}
			<div className="relative z-10 mx-auto max-w-5xl space-y-8">
				<header className="flex items-center justify-between rounded-xl border bg-background/60 px-6 py-5 backdrop-blur">
					<div>
						<h1 className="text-3xl font-bold tracking-tight">TeleopXR</h1>
						<p className="text-muted-foreground">
							Robot Teleoperation Dashboard
						</p>
					</div>
					<div className="flex items-center gap-3">
						{selectedMode === null ? (
							<>
								<Button
									size="lg"
									className="gap-2"
									onClick={handleEnterVR}
									variant="default"
								>
									<Monitor className="h-4 w-4" />
									VR Mode
								</Button>
								<Button
									size="lg"
									className="gap-2"
									onClick={handleEnterPassthrough}
									variant="secondary"
								>
									<Glasses className="h-4 w-4" />
									Passthrough
								</Button>
							</>
						) : (
							<Button
								size="lg"
								className="gap-2"
								onClick={handleExit}
								variant="destructive"
							>
								<X className="h-4 w-4" />
								Exit XR
							</Button>
						)}
					</div>
				</header>

				{xrError ? (
					<div className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm">
						<span className="font-medium text-destructive">XR:</span> {xrError}
					</div>
				) : null}

				<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
					<div className="space-y-6 lg:col-span-2">
						<TeleopPanel />
					</div>
					<div className="space-y-6">
						<CameraSettingsPanel />
						<AdvancedSettingsPanel />
					</div>
				</div>
			</div>
		</main>
	);
}
