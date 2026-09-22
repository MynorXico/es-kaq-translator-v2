interface AboutPageProps {
  onNavigateToTranslate: () => void;
}

const GITHUB_URL = "https://github.com/MynorXico/es-kaq-translator-v2";

export function AboutPage({ onNavigateToTranslate }: AboutPageProps) {
  return (
    <main className="about-page">
      <h1>Acerca de Traductor Kaqchikel</h1>
      <div className="zigzag" aria-hidden="true" />

      <section>
        <h2>Nuestra misión</h2>
        <p>
          Kaqchikel es un idioma maya que hablan cientos de miles de personas
          en Guatemala. Este proyecto existe para ayudar a preservarlo y
          facilitar la comunicación entre el español y el kaqchikel, no solo
          para construir una aplicación más.
        </p>
      </section>

      <section>
        <h2>¿De dónde vienen las traducciones?</h2>
        <p>
          El modelo aprendió de textos públicos proporcionados por la
          Academia de Lenguas Mayas de Guatemala (ALMG). Es un proyecto en
          desarrollo, así que la calidad de las traducciones todavía puede
          mejorar.
        </p>
      </section>

      <section>
        <h2>Ayúdanos a mejorar</h2>
        <p>
          Puedes proponer frases en español y kaqchikel, o avisarnos si una
          traducción no es correcta. Cada aporte ayuda a mejorar el
          traductor para toda la comunidad. El código de este proyecto
          también es abierto: cualquiera puede revisarlo en GitHub.
        </p>
        <a
          className="button button--outline"
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
        >
          Contribuir en GitHub
          <span className="sr-only"> (se abre en una pestaña nueva)</span>
        </a>
      </section>

      <p>
        <button type="button" className="link-button" onClick={onNavigateToTranslate}>
          ← Volver al traductor
        </button>
      </p>
    </main>
  );
}
