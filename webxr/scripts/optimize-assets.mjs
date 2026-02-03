import path from "node:path";
import { Logger, NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import {
	dedup,
	prune,
	quantize,
	textureCompress,
} from "@gltf-transform/functions";
import { parse } from "@pmndrs/uikitml";
import fs from "fs-extra";
import { glob } from "glob";
import sharp from "sharp";

const VERBOSE = true;

/**
 * GLTF Optimization Settings (Medium level)
 */
const GLTF_SETTINGS = {
	geometry: {
		precision: 0.8, // 0-1
	},
	textures: {
		quality: 0.75, // 0-1
		maxSize: 1024,
	},
};

/**
 * Helper to convert precision to quantization bits
 */
function toQuantizationBits(precision = 0.8) {
	precision = Math.max(0, Math.min(1, precision));
	return {
		quantizePosition: 10 + Math.round(precision * 6), // 10-16
		quantizeNormal: 8 + Math.round(precision * 2), // 8-10
		quantizeTexcoord: 10 + Math.round(precision * 2), // 10-12
		quantizeColor: 8,
		quantizeWeight: 8,
		quantizeGeneric: 10 + Math.round(precision * 2), // 10-12
	};
}

/**
 * Optimize GLB models
 */
async function optimizeModels() {
	const modelsDir = path.resolve(process.cwd(), "public/models");
	if (!(await fs.pathExists(modelsDir))) {
		if (VERBOSE)
			console.log(
				"Skipping model optimization: public/models directory not found",
			);
		return;
	}

	const files = await glob("**/*.glb", { cwd: modelsDir, absolute: true });
	if (files.length === 0) {
		if (VERBOSE) console.log("No GLB files found to optimize");
		return;
	}

	console.log(`Optimizing ${files.length} models...`);

	const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);

	for (const file of files) {
		try {
			const fileName = path.basename(file);
			if (VERBOSE) console.log(`🔄 Processing: ${fileName}`);

			const document = await io.read(file);

			if (VERBOSE) {
				document.setLogger(new Logger(Logger.Verbosity.INFO));
			}

			const quantizationBits = toQuantizationBits(
				GLTF_SETTINGS.geometry.precision,
			);

			await document.transform(
				prune(),
				dedup(),
				quantize(quantizationBits),
				textureCompress({
					encoder: sharp,
					resize: [
						GLTF_SETTINGS.textures.maxSize,
						GLTF_SETTINGS.textures.maxSize,
					],
				}),
			);

			await io.write(file, document);
			console.log(`✅ Optimized: ${fileName}`);
		} catch (error) {
			console.error(`❌ Failed to optimize ${file}:`, error);
		}
	}
}

/**
 * Compile UIKitML files
 */
async function compileUIKit() {
	const uiDir = path.resolve(process.cwd(), "ui");
	const outputDir = path.resolve(process.cwd(), "public/ui");

	if (!(await fs.pathExists(uiDir))) {
		if (VERBOSE)
			console.log("Skipping UIKitML compilation: ui directory not found");
		return;
	}

	const files = await glob("**/*.uikitml", { cwd: uiDir, absolute: true });
	if (files.length === 0) {
		if (VERBOSE) console.log("No UIKitML files found to compile");
		return;
	}

	console.log(`Compiling ${files.length} UIKitML files...`);
	await fs.ensureDir(outputDir);

	// Copy any existing JSON files from ui to public/ui first (as configs)
	const jsonFiles = await glob("**/*.json", { cwd: uiDir, absolute: true });
	for (const file of jsonFiles) {
		const relativePath = path.relative(uiDir, file);
		const outputPath = path.resolve(outputDir, relativePath);
		await fs.ensureDir(path.dirname(outputPath));
		await fs.copy(file, outputPath);
		if (VERBOSE) console.log(`📋 Copied config: ${path.basename(file)}`);
	}

	for (const file of files) {
		try {
			const fileName = path.basename(file);
			const relativePath = path.relative(uiDir, file);
			const outputPath = path.resolve(
				outputDir,
				relativePath.replace(/\.uikitml$/, ".json"),
			);

			if (VERBOSE) console.log(`🔄 Compiling: ${fileName}`);

			const sourceContent = await fs.readFile(file, "utf-8");
			const parseResult = parse(sourceContent, {
				onError: (message) => {
					console.error(
						`[compile-uikitml] Parse error in ${fileName}: ${message}`,
					);
				},
			});

			await fs.ensureDir(path.dirname(outputPath));
			await fs.writeFile(
				outputPath,
				JSON.stringify(parseResult, null, 2),
				"utf-8",
			);
			console.log(
				`✅ Compiled: ${fileName} -> ${path.relative(
					process.cwd(),
					outputPath,
				)}`,
			);
		} catch (error) {
			console.error(`❌ Failed to compile ${file}:`, error);
		}
	}
}

/**
 * Main
 */
async function main() {
	try {
		console.log("🚀 Starting asset optimization...");
		await optimizeModels();
		await compileUIKit();
		console.log("🏁 Asset optimization complete");
	} catch (error) {
		console.error("FATAL ERROR:", error);
		process.exit(1);
	}
}

main();
