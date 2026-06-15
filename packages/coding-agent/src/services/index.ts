/**
 * Whisper-Pi services for STT (speech-to-text) and TTS (text-to-speech).
 */

export {
	TypeWhisperAPI,
	TypeWhisperError,
	type DictationResult,
	type StatusResult,
	type ModelInfo,
	type TypeWhisperConfig,
	type TypeWhisperEngine,
} from "./typewhisper-api.ts";
export { STTService, type STTServiceConfig, type STTCallbacks, type STTState } from "./stt.ts";
export { TTSService, type TTSServiceConfig, type TTSCallbacks, type TTSState } from "./tts.ts";