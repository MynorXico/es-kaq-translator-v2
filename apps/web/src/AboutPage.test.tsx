import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { AboutPage } from "./AboutPage";

describe("AboutPage", () => {
  it("renders the three sections in order with the exact copy", () => {
    render(<AboutPage onNavigateToTranslate={vi.fn()} />);

    const headings = screen.getAllByRole("heading").map((heading) => heading.textContent);
    expect(headings).toEqual([
      "Acerca de Traductor Kaqchikel",
      "Nuestra misión",
      "¿De dónde vienen las traducciones?",
      "Ayúdanos a mejorar",
    ]);

    expect(
      screen.getByText(
        "Kaqchikel es un idioma maya que hablan cientos de miles de personas" +
          " en Guatemala. Este proyecto existe para ayudar a preservarlo y" +
          " facilitar la comunicación entre el español y el kaqchikel, no solo" +
          " para construir una aplicación más.",
        { normalizer: (text) => text.replace(/\s+/g, " ").trim() },
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "El modelo aprendió de textos públicos proporcionados por la" +
          " Academia de Lenguas Mayas de Guatemala (ALMG). Es un proyecto en" +
          " desarrollo, así que la calidad de las traducciones todavía puede" +
          " mejorar.",
        { normalizer: (text) => text.replace(/\s+/g, " ").trim() },
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "Puedes proponer frases en español y kaqchikel, o avisarnos si una" +
          " traducción no es correcta. Cada aporte ayuda a mejorar el" +
          " traductor para toda la comunidad. El código de este proyecto" +
          " también es abierto: cualquiera puede revisarlo en GitHub.",
        { normalizer: (text) => text.replace(/\s+/g, " ").trim() },
      ),
    ).toBeInTheDocument();
  });

  it("links to the GitHub repo, opening in a new tab, with an sr-only hint", () => {
    render(<AboutPage onNavigateToTranslate={vi.fn()} />);

    const link = screen.getByRole("link", { name: /Contribuir en GitHub/ });
    expect(link).toHaveAttribute("href", "https://github.com/MynorXico/es-kaq-translator-v2");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    expect(link).toHaveTextContent("(se abre en una pestaña nueva)");
  });

  it("calls onNavigateToTranslate when the bottom back link is clicked", () => {
    const onNavigateToTranslate = vi.fn();
    render(<AboutPage onNavigateToTranslate={onNavigateToTranslate} />);

    fireEvent.click(screen.getByRole("button", { name: "← Volver al traductor" }));

    expect(onNavigateToTranslate).toHaveBeenCalledTimes(1);
  });
});
