// Minimal stroke-icon placeholders (per #52's design spec). Decorative only
// (always aria-hidden): meaning is conveyed by a visible text label, or by
// an aria-label on the button itself for the one sanctioned icon-only
// exception below.
//
// Default pattern: icon + visible text label, never icon-only, so an
// infrequent digital-tool user isn't left guessing (see the Borrar/Copiar
// buttons, #38).
//
// Sanctioned exception: the direction-swap button (#39) is icon-only by
// design -- #52's spec places it as a compact control between the two
// direction labels, Google-Translate-style, with an `aria-label` carrying
// its accessible name instead of visible text.
import type { SVGProps } from "react";

function IconBase(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    />
  );
}

export function XCircleIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <IconBase {...props}>
      <circle cx="12" cy="12" r="9" />
      <line x1="9" y1="9" x2="15" y2="15" />
      <line x1="15" y1="9" x2="9" y2="15" />
    </IconBase>
  );
}

export function CopyIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <IconBase {...props}>
      <rect x="8" y="8" width="12" height="12" rx="2" />
      <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
    </IconBase>
  );
}

export function CheckIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <IconBase {...props}>
      <polyline points="20 6 9 17 4 12" />
    </IconBase>
  );
}

export function SwapIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <IconBase {...props}>
      <polyline points="7 4 3 8 7 12" />
      <line x1="3" y1="8" x2="21" y2="8" />
      <polyline points="17 12 21 16 17 20" />
      <line x1="21" y1="16" x2="3" y2="16" />
    </IconBase>
  );
}

// The wordmark's globe icon (#111): decorative, alongside the always-visible
// "Traductor Kaqchikel" text in the app-bar, so it stays aria-hidden like the
// other icons here rather than carrying its own accessible name.
export function GlobeIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <IconBase width={20} height={20} {...props}>
      <circle cx="12" cy="12" r="9" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <path d="M12 3c2.5 2.5 3.8 5.7 3.8 9s-1.3 6.5-3.8 9c-2.5-2.5-3.8-5.7-3.8-9S9.5 5.5 12 3Z" />
    </IconBase>
  );
}
