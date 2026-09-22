import { useEffect, useRef, useState } from "react";
import {
  translate,
  TranslateHttpError,
  TranslateTimeoutError,
  type TranslationDirection,
} from "./api";
import { AboutPage } from "./AboutPage";
import { CheckIcon, CopyIcon, SwapIcon, XCircleIcon } from "./icons";

const LANGUAGE_LABELS: Record<TranslationDirection, { source: string; target: string }> = {
  "es-to-cak": { source: "Español", target: "Kaqchikel" },
  "cak-to-es": { source: "Kaqchikel", target: "Español" },
};

// BCP-47-ish language tags for the `lang` attribute on the input/output
// textareas (WCAG 3.1.2 Language of Parts). "cak" is Kaqchikel's ISO 639-3
// code -- there's no ISO 639-1 two-letter code for it.
const LANGUAGE_TAGS: Record<TranslationDirection, { source: string; target: string }> = {
  "es-to-cak": { source: "es", target: "cak" },
  "cak-to-es": { source: "cak", target: "es" },
};

// Mirrors apps/api's TranslateRequest.text max_length (apps/api/app/models.py).
const MAX_INPUT_LENGTH = 2000;

// If the request is still in flight after this long, assume it's a
// Serverless Inference cold start (ADR 0001) and say so, rather than
// leaving the user looking at an indefinite spinner.
const WARMING_UP_DELAY_MS = 4000;

const OUTPUT_PLACEHOLDER = "La traducción aparecerá aquí.";

// How long the "Copiado"/"No se pudo copiar" confirmation replaces the
// "Copiar" label before reverting.
const COPY_FEEDBACK_MS = 2000;

type View = "translate" | "about";

type CopyState = "idle" | "success" | "error";

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
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const [announcement, setAnnouncement] = useState("");
  // Bumped by runTranslate (a new request starting) and by handleSwap/
  // handleClear (abandoning whatever request is in flight), so a response
  // for a request that's no longer current can't clobber later state.
  // Not touched by ordinary typing.
  const requestIdRef = useRef(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const outputRef = useRef<HTMLTextAreaElement>(null);
  const copyRevertTimerRef = useRef<ReturnType<typeof setTimeout>>();
  // Bumped (unconditionally) by handleSwap whenever it carries text into the
  // input, to trigger the caret-to-end effect below exactly once per swap.
  // Deliberately not keyed off `input` itself: if the carried-over
  // translation happens to equal the current input by value, setInput()
  // becomes a no-op React update that never changes `input`, which would
  // leave an `[input]`-keyed effect permanently "armed" to hijack focus/
  // caret on some later, unrelated edit.
  const [caretResetToken, setCaretResetToken] = useState(0);
  const isInitialCaretEffectRef = useRef(true);

  const isOverLimit = input.length > MAX_INPUT_LENGTH;
  const isTranslating = translateState.status === "loading" || translateState.status === "warming";

  useEffect(() => {
    if (isInitialCaretEffectRef.current) {
      isInitialCaretEffectRef.current = false;
      return;
    }
    const el = inputRef.current;
    if (el) {
      el.focus();
      el.setSelectionRange(el.value.length, el.value.length);
    }
  }, [caretResetToken]);

  // Focus management (WCAG 2.4.3): once a translation resolves, move focus
  // to the result so a screen reader user lands on it immediately, rather
  // than staying on the (now-disabled) "Traducir" button. Guarded on
  // `document.activeElement`: the input is never disabled while a
  // translation is in flight, so a user can start typing their *next*
  // query before this one resolves -- don't yank focus/caret out of the
  // input mid-sentence in that case (see issue #94).
  useEffect(() => {
    if (translateState.status === "success" && document.activeElement !== inputRef.current) {
      outputRef.current?.focus();
    }
  }, [translateState]);

  function handleSwap() {
    requestIdRef.current += 1;
    clearTimeout(copyRevertTimerRef.current);

    const nextDirection = direction === "es-to-cak" ? "cak-to-es" : "es-to-cak";
    const nextLabels = LANGUAGE_LABELS[nextDirection];

    if (translateState.status === "success") {
      setInput(translateState.translation);
      setCaretResetToken((token) => token + 1);
    }

    setDirection(nextDirection);
    setTranslateState({ status: "idle" });
    setCopyState("idle");
    setAnnouncement(`Dirección cambiada: ${nextLabels.source} a ${nextLabels.target}.`);
  }

  async function runTranslate() {
    const requestId = ++requestIdRef.current;
    setCopyState("idle");
    setTranslateState({ status: "loading" });
    const warmingTimer = setTimeout(() => {
      if (requestIdRef.current === requestId) {
        setTranslateState({ status: "warming" });
      }
    }, WARMING_UP_DELAY_MS);

    try {
      const result = await translate(input, direction);
      if (requestIdRef.current === requestId) {
        setTranslateState({ status: "success", translation: result.translation });
      }
    } catch (error) {
      if (requestIdRef.current === requestId) {
        setTranslateState({ status: "error", kind: classifyError(error) });
      }
    } finally {
      clearTimeout(warmingTimer);
    }
  }

  function handleClear() {
    requestIdRef.current += 1;
    clearTimeout(copyRevertTimerRef.current);
    setInput("");
    setTranslateState({ status: "idle" });
    setCopyState("idle");
    inputRef.current?.focus();
  }

  async function handleCopy() {
    if (translateState.status !== "success") {
      return;
    }
    clearTimeout(copyRevertTimerRef.current);
    try {
      await navigator.clipboard.writeText(translateState.translation);
      setCopyState("success");
    } catch {
      setCopyState("error");
    }
    copyRevertTimerRef.current = setTimeout(() => setCopyState("idle"), COPY_FEEDBACK_MS);
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

  const copyLabel =
    copyState === "success" ? "Copiado" : copyState === "error" ? "No se pudo copiar" : "Copiar";

  const { source: sourceLabel, target: targetLabel } = LANGUAGE_LABELS[direction];
  const { source: sourceLang, target: targetLang } = LANGUAGE_TAGS[direction];
  const inputAriaLabel = `Texto en ${sourceLabel.toLowerCase()}`;
  const outputAriaLabel = `Traducción en ${targetLabel.toLowerCase()}`;

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

        <div className="direction-row">
          <span className="direction-label direction-label--source">{sourceLabel}</span>
          <button
            type="button"
            className="icon-button"
            onClick={handleSwap}
            aria-label="Cambiar dirección"
          >
            <SwapIcon />
          </button>
          <span className="direction-label direction-label--target">{targetLabel}</span>
        </div>

        <textarea
          ref={inputRef}
          aria-label={inputAriaLabel}
          lang={sourceLang}
          placeholder="Escribe aquí..."
          value={input}
          onChange={(event) => setInput(event.target.value)}
        />

        <div className="input-meta-row">
          <p className={isOverLimit ? "char-counter char-counter--over-limit" : "char-counter"}>
            {input.length} / {MAX_INPUT_LENGTH}
          </p>

          {input.length > 0 && (
            <button
              type="button"
              className="text-button"
              onClick={handleClear}
              aria-label="Borrar el texto de entrada"
            >
              <XCircleIcon />
              Borrar
            </button>
          )}
        </div>

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

        <div className="output-card" role="status" aria-live="polite">
          <span className="sr-only">{announcement}</span>

          <div className="output-header">
            <p className="eyebrow">Traducción</p>

            {translateState.status === "success" && (
              <button type="button" className="text-button" onClick={handleCopy}>
                {copyState === "success" ? <CheckIcon /> : <CopyIcon />}
                {copyLabel}
              </button>
            )}
          </div>

          <div className="output-panel">
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
                <button
                  type="button"
                  className="button-outline"
                  onClick={runTranslate}
                  disabled={isTranslating}
                >
                  Reintentar
                </button>
              </div>
            ) : (
              <textarea
                ref={outputRef}
                aria-label={outputAriaLabel}
                lang={targetLang}
                placeholder={OUTPUT_PLACEHOLDER}
                value={translateState.status === "success" ? translateState.translation : ""}
                readOnly
              />
            )}
          </div>
        </div>
      </main>
    </>
  );
}
