"use client";

import { Rocket } from "lucide-react";
import dynamic from "next/dynamic";
import { useState } from "react";
import { CameraSettingsPanel } from "@/components/dashboard/CameraSettingsPanel";
import { TeleopPanel } from "@/components/dashboard/TeleopPanel";
import { Button } from "@/components/ui/button";

const XRScene = dynamic(
	() => import("@/components/xr/XRScene").then((mod) => mod.XRScene),
	{ ssr: false },
);

export default function Home() {
	const [isXRActive, setIsXRActive] = useState(false);

	return (
		<main className="min-h-screen bg-background p-8">
			{isXRActive && <XRScene onExit={() => setIsXRActive(false)} />}
			<div className="mx-auto max-w-5xl space-y-8">
				<header className="flex items-center justify-between border-b pb-6">
					<div>
						<h1 className="text-3xl font-bold tracking-tight">TeleopXR</h1>
						<p className="text-muted-foreground">
							Robot Teleoperation Dashboard
						</p>
					</div>
					<Button
						size="lg"
						className="gap-2"
						onClick={() => setIsXRActive(true)}
					>
						<Rocket className="h-4 w-4" />
						{isXRActive ? "XR Active" : "Launch XR"}
					</Button>
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
