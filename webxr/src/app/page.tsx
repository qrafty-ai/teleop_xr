"use client";

import { Rocket } from "lucide-react";
import dynamic from "next/dynamic";
import { useCallback, useState } from "react";
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
	const [xrError, setXrError] = useState<string | null>(null);
	const [xrBusy, setXrBusy] = useState(false);
	const isPassthroughEnabled = useAppStore(
		(state) => state.isPassthroughEnabled,
	);
	const isImmersiveActive = useAppStore((state) => state.isImmersiveActive);
	const setIsPassthroughEnabled = useAppStore(
		(state) => state.setIsPassthroughEnabled,
	);
	const xrActions = useAppStore((state) => state.xrActions);

	const handlePassthroughChange = useCallback(
		async (enabled: boolean) => {
			setIsPassthroughEnabled(enabled);
			if (!isImmersiveActive || xrBusy) return;
			if (!xrActions) {
				setXrError("XR controls are not ready yet");
				return;
			}
			setXrError(null);
			setXrBusy(true);
			try {
				await xrActions.enterXR({ passthrough: enabled });
			} catch (err) {
				setXrError(err instanceof Error ? err.message : "XR action failed");
			} finally {
				setXrBusy(false);
			}
		},
		[
			isImmersiveActive,
			setIsPassthroughEnabled,
			xrActions,
			xrBusy,
		],
	);

	const handleXrButtonClick = useCallback(async () => {
		if (xrBusy) return;
		setXrError(null);
		setXrBusy(true);
		try {
			if (isImmersiveActive) {
				if (!xrActions) {
					throw new Error("XR controls are not ready yet");
				}
				await xrActions.exitXR();
			} else {
				if (!xrActions) {
					throw new Error("XR controls are not ready yet");
				}
				await xrActions.enterXR({ passthrough: isPassthroughEnabled });
			}
		} catch (err) {
			setXrError(err instanceof Error ? err.message : "XR action failed");
		} finally {
			setXrBusy(false);
		}
	}, [isImmersiveActive, isPassthroughEnabled, xrActions, xrBusy]);

	return (
		<main className="min-h-screen bg-transparent p-8">
			<XRScene onError={(message) => setXrError(message)} />
			<div className="relative z-10 mx-auto max-w-5xl space-y-8">
				<header className="flex items-center justify-between rounded-xl border bg-background/60 px-6 py-5 backdrop-blur">
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
								onCheckedChange={handlePassthroughChange}
								disabled={xrBusy}
							/>
							<Label htmlFor="passthrough-mode">Passthrough</Label>
						</div>
						<Button
							size="lg"
							className="gap-2"
							onClick={handleXrButtonClick}
							variant={isImmersiveActive ? "destructive" : "default"}
							disabled={xrBusy}
						>
							<Rocket className="h-4 w-4" />
							{isImmersiveActive ? "Exit XR" : "Enter XR"}
						</Button>
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
					</div>
				</div>
			</div>
		</main>
	);
}
