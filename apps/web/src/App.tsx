import { useState } from "react";
import { translate, type TranslationDirection } from "./api";
import { AboutPage } from "./AboutPage";

const DIRECTION_LABELS: Record<TranslationDirection, string> = {
  "es-to-cak": "Español → Kaqchikel",
  "cak-to-es": "Kaqchikel → Español",
};

// Mirrors apps/api's TranslateRequest.text max_length (apps/api/app/models.py).
const MAX_INPUT_LENGTH = 2000;

type View = "translate" | "about";

export default function App() {
  const [view, setView] = useState<View>("translate");
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

  function toggleView() {
    setView((current) => (current === "translate" ? "about" : "translate"));
  }

  if (view === "about") {
    return (
      <>
        <header className="app-header">
          <button type="button" className="nav-link" onClick={toggleView}>
            Traducir
          </button>
        </header>
        <AboutPage onNavigateToTranslate={toggleView} />
      </>
    );
  }

  return (
    <>
      <header className="app-header">
        <button type="button" className="nav-link" onClick={toggleView}>
          Acerca de
        </button>
      </header>
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
    </>
  );
}
