export type TranslationDirection = "es-to-cak" | "cak-to-es";

export interface TranslateResult {
  translation: string;
}

/**
 * Placeholder client for the translation API. Swap this out once
 * apps/api exposes a real POST /v1/translate endpoint backed by the
 * fine-tuned model (see docs/adr/0001-initial-architecture.md).
 */
export async function translate(
  _text: string,
  _direction: TranslationDirection,
): Promise<TranslateResult> {
  return {
    translation: "Esta función estará disponible próximamente.",
  };
}
