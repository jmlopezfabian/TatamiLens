# tatami-web

Frontend de **TatamiLens**: sube videos de jiu-jitsu (BJJ), lanza análisis, y revisa/corrige las posiciones
y el marcador sobre una línea de tiempo sincronizada con el video.

Stack: React 19 + Vite + TypeScript (strict), react-router, TanStack Query, Tailwind CSS v4, Vitest, ESLint.
Contrato de la API: `../docs/CONTRACTS.md`.

## Arranque

Requiere Node 20+ y npm.

```bash
npm install
```

### Modo mock (sin backend)

Cliente en memoria con un análisis de ejemplo (~48 s: de pie → derribo → guardia cerrada → media guardia →
pase → side control → montada). No hay video real: el reproductor muestra un placeholder con reloj simulado.

```bash
npm run dev:mock        # equivale a VITE_USE_MOCK=true vite
```

Abre http://localhost:5173. Hay un video `demo_combate.mp4` ya analizado; subir un video nuevo y lanzar un
análisis simula cola → progreso → resultado en ~4 s. El "recalcular marcador" del mock usa una aproximación
local de las reglas IBJJF (`src/lib/rescore.ts`).

### Modo real

Con `tatami-api` corriendo en el puerto 8000 (CORS habilitado para `http://localhost:5173`):

```bash
npm run dev
```

### Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | URL base del backend (se le agrega `/api`) |
| `VITE_USE_MOCK` | `false` | `true` usa el cliente mock en memoria |

Ver `.env.example`; copia a `.env.local` para personalizar.

### Otros comandos

```bash
npm run build   # tsc + vite build
npm run lint    # ESLint
npm test        # Vitest (lógica pura de src/lib)
```

## Estructura

```
src/
  api/          cliente tipado: types.ts (contrato), client.ts (interfaz TatamiApi), real.ts (fetch/XHR),
                mock.ts + sample.ts (en memoria), index.ts (selección por env), hooks.ts (TanStack Query, polling)
  lib/          lógica pura con tests: score (marcador hasta t), segments (frames→segmentos),
                intervals (split/merge/mover bordes), rescore (eventos para el mock), positions, format
  components/   Player, Timeline, Scoreboard, EventList, AthletesPanel, UploadZone, AnalysisLauncher
  pages/        LibraryPage, VideoPage, LabelsPage
  hooks.ts      usePlayback (reloj unificado video real / simulado), useAthleteNames
```

## Notas

- Tema claro/oscuro: sigue el sistema y se puede cambiar con el botón (se guarda en localStorage).
- Los nombres de atletas se guardan en el navegador; las descripciones A/B van en las etiquetas (`PUT /labels`).
- "Recalcular marcador" guarda las etiquetas pendientes y luego llama `POST /api/analyses/{id}/rescore`.
