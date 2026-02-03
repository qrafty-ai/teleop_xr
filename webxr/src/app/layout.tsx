import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
	title: "TeleopXR",
	description: "WebXR Teleoperation Interface",
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang="en">
			<body
				className={cn(
					inter.className,
					"min-h-screen bg-background font-sans antialiased",
				)}
			>
				{children}
			</body>
		</html>
	);
}
