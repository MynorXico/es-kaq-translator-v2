import { useState } from "react";
import {
  translate,
  TranslateHttpError,
  TranslateTimeoutError,
  type TranslationDirection,
} from "./api";
import { AboutPage } from "./AboutPage";

const DIRECTION_LABELS: Record<TranslationDirection, string> = {
  "es-to-cak": "Español → Kaqchikel",
  "cak-to-es": "Kaqchikel → Español",
};

// Mirrors apps/api's TranslateRequest.text max_length (apps/api/app/models.py).
const MAX_INPUT_LENGTH = 2000;

// If the request is still in flight after this long, assume it's a
// Serverless Inference cold start (ADR 0001) and say so, rather than
// leaving the user looking at an indefinite spinner.
const WARMING_UP_DELAY_MS = 4000;

const OUTPUT_PLACEHOLDER = "La traducción aparecerá aquí.";

type View = "translate" | "about";

type TranslateErrorKind = "network" | "client" | "server" | "timeout";

type TranslateState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "warming" }
  | { status: "success"; translation: string }
  | { status: "error"; kind: TranslateErrorKind };

const ERROR_MESSAGES: Record<TranslateErrorKind, string> = {
  network: "No se pudo conectar. Revisa tu conexión a internet e inténtalo de nuevo.",
  client: "No se pudo traducir ese texto. Revisa lo que escribiste e inténtalo de nuevo.",
  server: "Hubo un problema en el servidor. Inténtalo de nuevo en unos minutos.",
  timeout: "El modelo está tardando más de lo esperado. Inténtalo de nuevo en unos minutos.",
};

function classifyError(error: unknown): TranslateErrorKind {
  if (error instanceof TranslateTimeoutError) {
    return "timeout";
  }
  if (error instanceof TranslateHttpError) {
    return error.status >= 500 ? "server" : "client";
  }
  return "network";
}

export default function App() {
  const [view, setView] = useState<View>("translate");
  const [direction, setDirection] = useState<TranslationDirection>("es-to-cak");
  const [input, setInput] = useState("");
  const [translateState, setTranslateState] = useState<TranslateState>({ status: "idle" });

  const isOverLimit = input.length > MAX_INPUT_LENGTH;
  const isTranslating = translateState.status === "loading" || translateState.status === "warming";

  function toggleDirection() {
    setDirection((current) => (current === "es-to-cak" ? "cak-to-es" : "es-to-cak"));
    setTranslateState({ status: "idle" });
  }

  async function runTranslate() {
    setTranslateState({ status: "loading" });
    const warmingTimer = setTimeout(() => {
      setTranslateState({ status: "warming" });
    }, WARMING_UP_DELAY_MS);

    try {
      const result = await translate(input, direction);
      setTranslateState({ status: "success", translation: result.translation });
    } catch (error) {
      setTranslateState({ status: "error", kind: classifyError(error) });
    } finally {
      clearTimeout(warmingTimer);
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

  const statusMessage =
    translateState.status === "loading"
      ? "Traduciendo…"
      : translateState.status === "warming"
        ? "Calentando el modelo. Puede tardar hasta un minuto."
        : translateState.status === "error"
          ? ERROR_MESSAGES[translateState.kind]
          : "";

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
          onClick={runTranslate}
          disabled={isTranslating || !input.trim() || isOverLimit}
        >
          Traducir
        </button>

        <div className="output-panel" role="status" aria-live="polite">
          {translateState.status === "loading" || translateState.status === "warming" ? (
            <div className="output-status">
              <span className="spinner" aria-hidden="true" />
              <span>{statusMessage}</span>
            </div>
          ) : translateState.status === "error" ? (
            <div className="output-status output-status--error">
              <p className="output-error-message">
                <span aria-hidden="true">⚠</span> <span>{statusMessage}</span>
              </p>
              <button type="button" className="button-outline" onClick={runTranslate}>
                Reintentar
              </button>
            </div>
          ) : (
            <textarea
              aria-label="Traducción"
              placeholder={OUTPUT_PLACEHOLDER}
              value={translateState.status === "success" ? translateState.translation : ""}
              readOnly
            />
          )}
        </div>
      </main>
    </>
  );
}
