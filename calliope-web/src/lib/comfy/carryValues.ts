import { normalizeInputRole } from './parser';
import type { ComfyDynamicInput } from './types';

/**
 * Workflow form values are keyed by nodeId, and a nodeId only means something inside ONE
 * workflow: node "3" is Width in "AI1 · MiniMax H3 ref r2v" but FILM interpolation in
 * "AI1 · VIDEO FINISH". Carrying values across a workflow switch unchecked submitted the
 * finish lane's `3: 2, 4: 2, 5: "ultra"` as a 2×2 frame with seed "ultra", which the rig
 * rejects (width/height must be >= 256), and auto-save persisted it on the clip.
 */
export type WorkflowValues = Record<string, string | number>;

/** No served lane renders below 256 px; anything under this is a stale non-dimension value. */
export const MIN_DIMENSION_PX = 64;

function signature(inp: ComfyDynamicInput): string {
	return `${normalizeInputRole(inp.role ?? null) ?? ''}|${inp.kind}`;
}

/**
 * Values to keep when switching from one workflow's schema to another: only where the same
 * nodeId carries the same role and kind in both. Everything else is dropped so the new
 * workflow's defaults apply (the composer prefills undefined fields).
 */
export function carryValuesAcrossWorkflows(
	values: WorkflowValues,
	from: ComfyDynamicInput[] | undefined,
	to: ComfyDynamicInput[] | undefined,
): WorkflowValues {
	const fromSig = new Map((from ?? []).map((inp) => [inp.nodeId, signature(inp)] as const));
	const out: WorkflowValues = {};
	for (const inp of to ?? []) {
		const v = values[inp.nodeId];
		if (v === undefined || v === '') continue;
		if (fromSig.get(inp.nodeId) === signature(inp)) out[inp.nodeId] = v;
	}
	return out;
}

function isInvalidFor(inp: ComfyDynamicInput, v: string | number): boolean {
	if (inp.kind !== 'number' || v === '') return false;
	const n = typeof v === 'number' ? v : Number(v);
	if (!Number.isFinite(n)) return true;
	const role = normalizeInputRole(inp.role ?? null);
	return (role === 'width' || role === 'height') && n < MIN_DIMENSION_PX;
}

/**
 * Drop saved values that cannot be right for this workflow (a non-numeric value in a number
 * input, or a width/height below MIN_DIMENSION_PX) so defaults apply again. Heals clip setups
 * persisted before workflow switches were remapped. Keys outside the schema pass through.
 */
export function sanitizeWorkflowValues(
	values: WorkflowValues,
	schema: ComfyDynamicInput[] | undefined,
): WorkflowValues {
	const byNode = new Map((schema ?? []).map((inp) => [inp.nodeId, inp] as const));
	const out: WorkflowValues = {};
	for (const [k, v] of Object.entries(values)) {
		const inp = byNode.get(k);
		if (inp && isInvalidFor(inp, v)) continue;
		out[k] = v;
	}
	return out;
}
