/**
 * Whisper-Pi services for STT (speech-to-text) and TTS (text-to-speech).
 */

export { RegexT5Parser } from "./regex-t5-parser.ts";
export { DEFAULT_PARSER_CONFIG, type ParserConfig, type ParserProvider } from "./parser-provider.ts";
export { type STTCallbacks, STTService, type STTServiceConfig, type STTState } from "./stt.ts";
export { type TTSCallbacks, TTSService, type TTSServiceConfig, type TTSState } from "./tts.ts";
export {
	DEFAULT_TTS_POLISHER_CONFIG,
	RegexTTSPolisher,
	type TTSPolisherConfig,
	type TTSPolisherProvider,
} from "./tts-polisher.ts";
export {
	type DictationResult,
	type ModelInfo,
	type StatusResult,
	TypeWhisperAPI,
	type TypeWhisperConfig,
	type TypeWhisperEngine,
	TypeWhisperError,
} from "./typewhisper-api.ts";
