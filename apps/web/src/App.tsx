import { useState } from "react";
import { translate, type TranslationDirection } from "./api";

const DIRECTION_LABELS: Record<TranslationDirection, string> = {
  "es-to-cak": "Español → Kaqchikel",
  "cak-to-es": "Kaqchikel → Español",
};

// Mirrors apps/api's TranslateRequest.text max_length (apps/api/app/models.py).
const MAX_INPUT_LENGTH = 2000;

export default function App() {
  const [direction, setDirection] = useState<TranslationDirection>("es-to-cak");
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);

  const isOverLimit = input.length > MAX_INPUT_LENGTH;

  function toggleDirection() {
    setDirection((current) => (current === "es-to-cak" ? "cak-to-es" : "es-to-cak"));
    setOutput("");
  }

  async function handleTranslate() {
    setIsTranslating(true);
    try {
      const result = await translate(input, direction);
      setOutput(result.translation);
    } finally {
      setIsTranslating(false);
    }
  }

  return (
    <main>
      <h1>Traductor Kaqchikel</h1>
      <p>Traducción español ↔ kaqchikel</p>

      <button type="button" onClick={toggleDirection}>
        {DIRECTION_LABELS[direction]}
      </button>

      <textarea
        aria-label="Texto a traducir"
        placeholder="Escribe aquí..."
        value={input}
        onChange={(event) => setInput(event.target.value)}
      />

      <p className={isOverLimit ? "char-counter char-counter--over-limit" : "char-counter"}>
        {input.length} / {MAX_INPUT_LENGTH}
      </p>

      {isOverLimit && (
        <p role="alert" className="over-limit-message">
          El texto supera el límite de {MAX_INPUT_LENGTH} caracteres.
        </p>
      )}

      <button
        type="button"
        onClick={handleTranslate}
        disabled={isTranslating || !input.trim() || isOverLimit}
      >
        {isTranslating ? "Traduciendo..." : "Traducir"}
      </button>

      <textarea
        aria-label="Traducción"
        placeholder="La traducción aparecerá aquí"
        value={output}
        readOnly
      />
    </main>
  );
}
