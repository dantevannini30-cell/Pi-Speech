import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { Container, type SelectItem, SelectList, type SelectListLayoutOptions } from "@earendil-works/pi-tui";
import { getSelectListTheme } from "../theme/theme.ts";
import { DynamicBorder } from "./dynamic-border.ts";

const VOICE_SELECT_LIST_LAYOUT: SelectListLayoutOptions = {
	minPrimaryColumnWidth: 20,
	maxPrimaryColumnWidth: 44,
};

const KOKORO_VOICES: SelectItem[] = [
	{ value: "kokoro:af_heart", label: "af_heart", description: "Kokoro" },
	{ value: "kokoro:af_bella", label: "af_bella", description: "Kokoro" },
	{ value: "kokoro:af_nicole", label: "af_nicole", description: "Kokoro" },
	{ value: "kokoro:af_sarah", label: "af_sarah", description: "Kokoro" },
	{ value: "kokoro:af_sky", label: "af_sky", description: "Kokoro" },
	{ value: "kokoro:am_adam", label: "am_adam", description: "Kokoro" },
	{ value: "kokoro:am_michael", label: "am_michael", description: "Kokoro" },
	{ value: "kokoro:am_fenrir", label: "am_fenrir", description: "Kokoro" },
	{ value: "kokoro:bf_emma", label: "bf_emma", description: "Kokoro" },
	{ value: "kokoro:bf_isabella", label: "bf_isabella", description: "Kokoro" },
	{ value: "kokoro:bm_george", label: "bm_george", description: "Kokoro" },
	{ value: "kokoro:bm_fable", label: "bm_fable", description: "Kokoro" },
];

function getPiperVoiceItems(): SelectItem[] {
	const __filename = fileURLToPath(import.meta.url);
	const __dirname = path.dirname(__filename);
	const piperDir = path.resolve(__dirname, "..", "..", "..", "..", "piper-models");
	try {
		const names = fs
			.readdirSync(piperDir)
			.filter((f: string) => f.endsWith(".onnx"))
			.map((f: string) => f.replace(/\.onnx$/, ""));
		return names.map((name) => ({
			value: `piper:${name}`,
			label: name,
			description: "Piper",
		}));
	} catch {
		return [];
	}
}

function getAllVoiceItems(currentVoice: string): SelectItem[] {
	const piperItems = getPiperVoiceItems();
	const all = [...piperItems, ...KOKORO_VOICES];
	// Mark current voice: match against the voice part of compound values
	return all.map((item) => {
		const [, voice] = item.value.split(":");
		const isCurrent = voice === currentVoice;
		return {
			...item,
			description: isCurrent ? `(current) ${item.description}` : item.description,
		};
	});
}

/**
 * Component that renders a TTS voice selector showing all available voices
 * from both Piper (scanned from disk) and Kokoro (built-in).
 */
export class TTSVoiceSelectorComponent extends Container {
	private selectList: SelectList;

	constructor(currentVoice: string, onSelect: (voice: string) => void, onCancel: () => void) {
		super();

		const items = getAllVoiceItems(currentVoice);

		this.addChild(new DynamicBorder());

		this.selectList = new SelectList(items, 10, getSelectListTheme(), VOICE_SELECT_LIST_LAYOUT);

		// Match current voice against the voice part of compound values
		const currentIndex = items.findIndex((item) => {
			const [, voice] = item.value.split(":");
			return voice === currentVoice;
		});
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
