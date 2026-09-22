/**
 * Primitive + character three.js object factory — Svelte port of
 * open-media's PrimitiveObject.tsx shape switch (MIT). Characters come from
 * mannequinFactory; primitives are plain three.js geometry with a neutral
 * blockout material.
 */
import * as THREE from 'three';
import type { PrimitiveType } from '../shotStore.svelte';

const BLOCKOUT_MATERIAL = new THREE.MeshStandardMaterial({
	color: 0x9aa4b2,
	roughness: 0.85,
	metalness: 0.05,
});

export function createPrimitiveMesh(type: PrimitiveType): THREE.Mesh {
	let geometry: THREE.BufferGeometry;
	switch (type) {
		case 'cube':
			geometry = new THREE.BoxGeometry(1, 1, 1);
			break;
		case 'plane':
			geometry = new THREE.BoxGeometry(4, 0.1, 4);
			break;
		case 'cylinder':
			geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 24);
			break;
		case 'sphere':
			geometry = new THREE.SphereGeometry(0.5, 24, 16);
			break;
		case 'capsule':
			geometry = new THREE.CapsuleGeometry(0.4, 1, 6, 16);
			break;
		case 'cone':
			geometry = new THREE.ConeGeometry(0.5, 1, 24);
			break;
		case 'torus':
			geometry = new THREE.TorusGeometry(0.5, 0.18, 16, 40);
			break;
	}
	const mesh = new THREE.Mesh(geometry, BLOCKOUT_MATERIAL);
	mesh.castShadow = true;
	mesh.receiveShadow = true;
	return mesh;
}
