import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  it("renders the translator title and direction toggle", () => {
    render(<App />);
    expect(screen.getByText("Traductor Kaqchikel")).toBeInTheDocument();
    expect(screen.getByText("Español → Kaqchikel")).toBeInTheDocument();
  });

  it("shows a character counter that updates as the user types", () => {
    render(<App />);
    const input = screen.getByLabelText("Texto a traducir");

    expect(screen.getByText("0 / 2000")).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "Hola" } });

    expect(screen.getByText("4 / 2000")).toBeInTheDocument();
  });

  it("disables translation and explains why once the input exceeds 2000 characters", () => {
    render(<App />);
    const input = screen.getByLabelText("Texto a traducir");
    const translateButton = screen.getByRole("button", { name: "Traducir" });

    fireEvent.change(input, { target: { value: "a".repeat(2001) } });

    expect(screen.getByText("2001 / 2000")).toBeInTheDocument();
    expect(
      screen.getByText("El texto supera el límite de 2000 caracteres."),
    ).toBeInTheDocument();
    expect(translateButton).toBeDisabled();
  });

  it("navigates to the About page via the header nav link, and back via the same slot", () => {
    render(<App />);

    const aboutLink = screen.getByRole("button", { name: "Acerca de" });
    fireEvent.click(aboutLink);

    expect(
      screen.getByRole("heading", { name: "Acerca de Traductor Kaqchikel" }),
    ).toBeInTheDocument();

    const translateLink = screen.getByRole("button", { name: "Traducir" });
    fireEvent.click(translateLink);

    expect(screen.getByText("Español → Kaqchikel")).toBeInTheDocument();
  });
});
