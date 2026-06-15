import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { Container, type SelectItem, SelectList, type SelectListLayoutOptions } from "@earendil-works/pi-tui";
import { getSelectListTheme } from "../theme/theme.ts";
import { DynamicBorder } from "./dynamic-border.ts";

const VOICE_SELECT_LIST_LAYOUT: SelectListLayoutOptions = {
	minPrimaryColumnWidth: 16,
	maxPrimaryColumnWidth: 40,
};

const KOKORO_VOICES: string[] = [
	"af_heart",
	"af_bella",
	"af_nicole",
	"af_sarah",
	"af_sky",
	"am_adam",
	"am_michael",
	"am_fenrir",
	"bf_emma",
	"bf_isabella",
	"bm_george",
	"bm_fable",
];

function getPiperVoices(): string[] {
	const __filename = fileURLToPath(import.meta.url);
	const __dirname = path.dirname(__filename);
	const piperDir = path.resolve(__dirname, "..", "..", "..", "..", "piper-models");
	try {
		return fs
			.readdirSync(piperDir)
			.filter((f: string) => f.endsWith(".onnx"))
			.map((f: string) => f.replace(/\.onnx$/, ""));
	} catch {
		return [];
	}
}

/**
 * Component that renders a TTS voice selector.
 */
export class TTSVoiceSelectorComponent extends Container {
	private selectList: SelectList;

	constructor(
		currentProvider: "piper" | "kokoro",
		currentVoice: string,
		onSelect: (voice: string) => void,
		onCancel: () => void,
	) {
		super();

		const voices = currentProvider === "piper" ? getPiperVoices() : KOKORO_VOICES;

		const items: SelectItem[] = voices.map((name) => ({
			value: name,
			label: name,
			description: name === currentVoice ? "(current)" : undefined,
		}));

		this.addChild(new DynamicBorder());

		this.selectList = new SelectList(items, 10, getSelectListTheme(), VOICE_SELECT_LIST_LAYOUT);

		const currentIndex = voices.indexOf(currentVoice);
		if (currentIndex !== -1) {
			this.selectList.setSelectedIndex(currentIndex);
		}

		this.selectList.onSelect = (item) => {
			onSelect(item.value);
		};

		this.selectList.onCancel = () => {
			onCancel();
		};

		this.addChild(this.selectList);
		this.addChild(new DynamicBorder());
	}

	getSelectList(): SelectList {
		return this.selectList;
	}
}
