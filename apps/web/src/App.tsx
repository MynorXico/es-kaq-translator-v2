import { useEffect, useRef, useState } from "react";
import {
  translate,
  TranslateHttpError,
  TranslateTimeoutError,
  type TranslationDirection,
} from "./api";
import { AboutPage } from "./AboutPage";
import { CheckIcon, CopyIcon, GlobeIcon, SwapIcon, XCircleIcon } from "./icons";

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

const APP_NAME = "Traductor Kaqchikel";

// Always shown alongside a successful translation (issue #43): the model is
// fine-tuned on a low-resource language and won't be perfect at launch, so
// results shouldn't be presented without a caveat. Deliberately generic
// about "Kaqchikel" -- the training corpus mixes two unlabeled dialectal
// variants, so no variant-specific claim can be made honestly (see #51/#90).
const QUALITY_DISCLAIMER =
  "Esta traducción fue generada automáticamente por un modelo en desarrollo y puede contener errores.";

// Issue #42: a link next to a successful translation lets a user report it
// as incorrect without needing to already know this repo exists. It opens a
// new GitHub issue pre-filled against the "Translation quality report"
// template (.github/ISSUE_TEMPLATE/translation_quality.md), so the reporter
// only has to add the expected translation and their background.
const REPO_URL = "https://github.com/MynorXico/es-kaq-translator-v2";

// Mirrors the checkbox options under the template's "## Direction" heading
// literally (including the English wording) so the pre-filled body matches
// the template's own text.
const DIRECTION_SECTION: Record<TranslationDirection, string> = {
  "es-to-cak": "- [x] Spanish -> Kaqchikel\n- [ ] Kaqchikel -> Spanish",
  "cak-to-es": "- [ ] Spanish -> Kaqchikel\n- [x] Kaqchikel -> Spanish",
};

// Builds a GitHub "new issue" URL targeting translation_quality.md, with the
// direction, input, and output already populated under the template's own
// section headings, so a query-string `body` param lands in the right
// places instead of just appending free text.
function buildTranslationReportUrl(
  direction: TranslationDirection,
  sourceText: string,
  translation: string,
): string {
  const body = [
    "## Direction",
    "",
    DIRECTION_SECTION[direction],
    "",
    "## Input text",
    "",
    sourceText,
    "",
    "## Output produced by the translator",
    "",
    translation,
    "",
    "## Expected / correct translation",
    "",
    "",
    "## Your background (optional, helps us weigh the report)",
    "",
    "- [ ] Native Kaqchikel speaker",
    "- [ ] Kaqchikel learner / student",
    "- [ ] Linguist / ALMG-affiliated",
    "- [ ] Other",
    "",
    "## Additional context",
    "",
  ].join("\n");

  const params = new URLSearchParams({
    template: "translation_quality.md",
    labels: "translation-quality",
    title: "[Translation] ",
    body,
  });

  return `${REPO_URL}/issues/new?${params.toString()}`;
}

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
  | { status: "success"; translation: string; sourceText: string }
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

// Decorative marigold/violet/blue top stripe (#52's design spec).
function Stripe() {
  return (
    <div className="stripe" aria-hidden="true">
      <span className="b1" />
      <span className="b2" />
      <span className="b3" />
    </div>
  );
}

// Shared app-bar for both the translator and About views: wordmark + globe
// icon on the left, a single nav link (whose label/handler flips depending
// on which view is currently showing) on the right.
function AppBar({ navLabel, onNavClick }: { navLabel: string; onNavClick: () => void }) {
  return (
    <header className="appbar">
      <span className="wordmark">
        <GlobeIcon />
        {APP_NAME}
      </span>
      <button type="button" className="nav-link" onClick={onNavClick}>
        {navLabel}
      </button>
    </header>
  );
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
    // Snapshotted now, not read from `input` at success time: the user can
    // keep editing the input while this request is in flight (see the
    // focus-management comment above), so `input` itself may no longer
    // match what was actually sent by the time the response comes back.
    // The report-a-bad-translation link (#42) needs the text that actually
    // produced the shown translation.
    const sourceText = input;
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
        setTranslateState({ status: "success", translation: result.translation, sourceText });
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
      <div className="app-shell">
        <Stripe />
        <AppBar navLabel="Traducir" onNavClick={toggleView} />
        <AboutPage onNavigateToTranslate={toggleView} />
      </div>
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
    <div className="app-shell">
      <Stripe />
      <AppBar navLabel="Acerca de" onNavClick={toggleView} />
      <main className="app-body">
        <h1>{APP_NAME}</h1>
        <p>Traducción español ↔ kaqchikel</p>

        <div className="direction-switch">
          <span className="dir-label is-source">{sourceLabel}</span>
          <button
            type="button"
            className="swap-btn"
            onClick={handleSwap}
            aria-label="Cambiar dirección"
          >
            <SwapIcon />
          </button>
          <span className="dir-label">{targetLabel}</span>
        </div>

        <div className="field-card">
          <div className="field-label-row">
            <span className="field-eyebrow">{inputAriaLabel}</span>
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
        </div>

        <button
          type="button"
          className="cta-button button-outline"
          onClick={runTranslate}
          disabled={isTranslating || !input.trim() || isOverLimit}
        >
          Traducir
        </button>

        <div className="field-card" role="status" aria-live="polite">
          <span className="sr-only">{announcement}</span>

          <div className="field-label-row">
            <span className="field-eyebrow">Traducción</span>

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

          {translateState.status === "success" && (
            <p className="output-disclaimer">{QUALITY_DISCLAIMER}</p>
          )}

          {translateState.status === "success" && (
            <a
              className="link-button"
              href={buildTranslationReportUrl(
                direction,
                translateState.sourceText,
                translateState.translation,
              )}
              target="_blank"
              rel="noopener noreferrer"
            >
              ¿Traducción incorrecta? Repórtala
              <span className="sr-only"> (se abre en una pestaña nueva)</span>
            </a>
          )}
        </div>
      </main>
    </div>
  );
}
