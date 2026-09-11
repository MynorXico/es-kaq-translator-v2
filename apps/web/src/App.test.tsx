import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  it("renders the translator title and direction toggle", () => {
    render(<App />);
    expect(screen.getByText("Traductor Kaqchikel")).toBeInTheDocument();
    expect(screen.getByText("Español → Kaqchikel")).toBeInTheDocument();
  });
});
