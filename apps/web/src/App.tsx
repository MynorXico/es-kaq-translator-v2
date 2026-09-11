import { useState } from "react";
import { translate, type TranslationDirection } from "./api";

const DIRECTION_LABELS: Record<TranslationDirection, string> = {
  "es-to-cak": "Español → Kaqchikel",
  "cak-to-es": "Kaqchikel → Español",
};

export default function App() {
  const [direction, setDirection] = useState<TranslationDirection>("es-to-cak");
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);

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

      <button type="button" onClick={handleTranslate} disabled={isTranslating || !input.trim()}>
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
