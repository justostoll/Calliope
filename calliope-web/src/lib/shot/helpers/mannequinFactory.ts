import * as THREE from "three";

export type CharacterType = "male" | "female" | "child";

/** Reused so grounding, which runs per animation frame during a motion, allocates nothing. */
const groundBox = new THREE.Box3();
const groundScale = new THREE.Vector3();

/**
 * Plants the figure's lowest point on y = 0. The one grounding routine — nothing
 * in this app may call mannequin-js's own `stepOnGround()`.
 *
 * `stepOnGround()` is wrong here twice over. It lands the feet on
 * `GROUND_LEVEL = -0.7`, the height of the ground disc in mannequin-js's
 * built-in Stage, which this app does not use — its grid is at y = 0, so every
 * call sank the figure 0.7, about 40% of a 1.8-unit body, taking it under the
 * grid and under the invisible deselect plane. And it writes a *world*-space
 * measurement straight into a *local* `position.y`, which is only the same thing
 * when the figure is a direct child of the scene, as it is in the prototype.
 * Here every figure hangs off a wrapper Group carrying the object transform, so
 * the wrapper's own Y was silently cancelled out.
 *
 * Same three steps as the original — clear the offset, measure, re-offset — with
 * the measurement converted back through the parent's world Y scale. The 0.01
 * sink is mannequin-js's own, and keeps the soles out of the grid plane.
 */
export function groundFigure(figure: THREE.Object3D): void {
  figure.position.y = 0;
  figure.updateWorldMatrix(true, true);

  groundBox.setFromObject(figure);
  if (groundBox.isEmpty()) return;

  const scaleY = figure.parent?.getWorldScale(groundScale).y || 1;
  figure.position.y = (-0.01 - groundBox.min.y) / scaleY;
  figure.updateWorldMatrix(true, true);
}

export interface CharacterOptions {
  type?: CharacterType;
}

/**
 * mannequin-js's scene.js runs `initStage()` as a module-load side effect:
 * a full-screen `position:fixed` WebGLRenderer canvas appended to
 * document.body, with a `gainsboro` background and an infinite render loop.
 * Importing ANY body class (Male/Female/Child) evaluates that module first —
 * the canvas then sits on top of every page (the "white flash", and a
 * permanent white overlay whenever anything threw before the old cleanup
 * ran). Its render loop also steals context/frame budget from the real
 * viewport.
 *
 * The fix is ordering: import scene.js OURSELVES first and tear the stage
 * down in the same task, before any body class is imported. The body
 * constructors only need the module's `scene` (a plain THREE.Scene) —
 * the stage's renderer/canvas are dead weight. Afterwards the stray canvas
 * never survives a paint, and every createMannequin sweeps again as
 * belt-and-braces (covers dev HMR re-evaluating the module).
 */
type StageRenderer = {
  setAnimationLoop(cb: unknown): void;
  dispose(): void;
};

interface MannequinSceneModule {
  getStage(): { renderer: StageRenderer } | undefined;
  scene: THREE.Scene;
}

let sceneModule: Promise<MannequinSceneModule> | null = null;

/** Removes the fixed-position <canvas> elements mannequin-js appends to document.body. */
export function removeMannequinCanvases(): void {
  if (typeof document !== "undefined") {
    document.querySelectorAll("body > canvas").forEach((c) => c.remove());
  }
}

async function loadSceneModule(): Promise<MannequinSceneModule> {
  const mod = (await import("mannequin-js/src/scene.js")) as MannequinSceneModule;
  // Same-task teardown: the canvas is appended during the import above and
  // removed below before the browser gets a rendering opportunity, so it
  // never paints.
  try {
    const stage = mod.getStage();
    stage?.renderer?.setAnimationLoop(null);
    stage?.renderer?.dispose();
  } catch {
    // a broken stage must not break figure creation
  }
  removeMannequinCanvases();
  return mod;
}

/** Resolves once the mannequin scene module exists AND its stray stage is dead. */
export function ensureMannequinStageSanitized(): Promise<MannequinSceneModule> {
  sceneModule ??= loadSceneModule();
  return sceneModule;
}

async function loadMannequin(type: CharacterType): Promise<THREE.Object3D> {
  switch (type) {
    case "male": {
      const { Male } = await import("mannequin-js/src/bodies/Male.js");
      return new Male() as unknown as THREE.Object3D;
    }
    case "female": {
      const { Female } = await import("mannequin-js/src/bodies/Female.js");
      return new Female() as unknown as THREE.Object3D;
    }
    case "child": {
      const { Child } = await import("mannequin-js/src/bodies/Child.js");
      return new Child() as unknown as THREE.Object3D;
    }
    default: {
      const { Male } = await import("mannequin-js/src/bodies/Male.js");
      return new Male() as unknown as THREE.Object3D;
    }
  }
}

export function createMannequin(options: CharacterOptions = {}): Promise<THREE.Object3D> {
  return (async () => {
    // Sanitize BEFORE the body-class import graph evaluates scene.js.
    await ensureMannequinStageSanitized();

    const { type = "male" } = options;
    const mannequin = await loadMannequin(type);

    // Remove the mannequin from any parent created internally.
    mannequin.parent?.remove(mannequin);

    // Sweep again — dev HMR may have re-evaluated scene.js since last time.
    removeMannequinCanvases();

    // The constructor ends on stepOnGround(), which left the figure at -0.71.
    groundFigure(mannequin);

    return mannequin;
  })();
}
