import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { DataDot } from '../types/ml';
import {
  Maximize2,
  Minimize2,
  RotateCcw,
  Sliders,
  ChevronDown,
  ChevronUp,
  Layers,
  Sparkles,
  Compass,
  Play,
  Pause,
  Box,
} from 'lucide-react';

interface ThreeCosmosSceneProps {
  dots?: DataDot[];
  customDatasetName?: string;
  onOpenIngestModal?: () => void;
}

type SceneMode = 'loss-bowl' | 'svm-hyperplane' | 'cluster-space' | 'neural-terrain';

// Module-level pre-allocated scratch objects to prevent GC in render loop
const _vec3 = new THREE.Vector3();
const _mat4 = new THREE.Matrix4();
const _color = new THREE.Color();
const _quat = new THREE.Quaternion();

export const ThreeCosmosScene: React.FC<ThreeCosmosSceneProps> = ({
  dots = [],
  customDatasetName,
  onOpenIngestModal,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [sceneMode, setSceneMode] = useState<SceneMode>('loss-bowl');
  const [isControlsCollapsed, setIsControlsCollapsed] = useState<boolean>(false);
  const [isAutoRotate, setIsAutoRotate] = useState<boolean>(true);
  const [surfaceResolution, setSurfaceResolution] = useState<number>(40);
  const [surfaceSteepness, setSurfaceSteepness] = useState<number>(1.2);
  const [showWireframe, setShowWireframe] = useState<boolean>(false);
  const [animSpeed, setAnimSpeed] = useState<number>(1);

  // References to mutable Three.js objects
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const instancedDotsRef = useRef<THREE.InstancedMesh | null>(null);
  const surfaceMeshRef = useRef<THREE.Mesh | null>(null);
  const heroGroupRef = useRef<THREE.Group | null>(null);
  const descentBallRef = useRef<THREE.Mesh | null>(null);
  const frameIdRef = useRef<number>(0);

  // Keep state sync in refs for 60fps loop without re-running effect
  const modeRef = useRef<SceneMode>(sceneMode);
  const autoRotateRef = useRef<boolean>(isAutoRotate);
  const steepnessRef = useRef<number>(surfaceSteepness);
  const animSpeedRef = useRef<number>(animSpeed);

  useEffect(() => {
    modeRef.current = sceneMode;
  }, [sceneMode]);

  useEffect(() => {
    autoRotateRef.current = isAutoRotate;
  }, [isAutoRotate]);

  useEffect(() => {
    steepnessRef.current = surfaceSteepness;
  }, [surfaceSteepness]);

  useEffect(() => {
    animSpeedRef.current = animSpeed;
  }, [animSpeed]);

  // Main Three.js Scene Setup
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    // 1. Detect Software WebGL
    const isSoftwareGl = (() => {
      const gl = document.createElement('canvas').getContext('webgl2');
      if (!gl) return true;
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
      const gpu = String(gl.getParameter(param));
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();

    // 2. Initialize Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
      alpha: false,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 3. Scene & Horizon Fog
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617); // Slate-950
    scene.fog = new THREE.FogExp2(0x020617, 0.022);
    sceneRef.current = scene;

    // 4. Studio Environment Lighting
    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
    pmremGenerator.dispose();

    // 5. Studio Key & Fill Lighting
    const keyLight = new THREE.DirectionalLight(0xffeedd, 2.2);
    keyLight.position.set(15, 25, 12);
    keyLight.castShadow = !isSoftwareGl;
    if (keyLight.castShadow) {
      keyLight.shadow.mapSize.width = 1024;
      keyLight.shadow.mapSize.height = 1024;
      keyLight.shadow.camera.near = 0.5;
      keyLight.shadow.camera.far = 60;
      keyLight.shadow.camera.left = -15;
      keyLight.shadow.camera.right = 15;
      keyLight.shadow.camera.top = 15;
      keyLight.shadow.camera.bottom = -15;
      keyLight.shadow.bias = -0.0005;
      keyLight.shadow.normalBias = 0.02;
    }
    scene.add(keyLight);

    const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x0f172a, 1.2);
    scene.add(hemiLight);

    const accentRimLight = new THREE.PointLight(0xa855f7, 3.5, 40);
    accentRimLight.position.set(-18, 12, -15);
    scene.add(accentRimLight);

    // 6. Camera & Damped OrbitControls
    const camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 100);
    camera.position.set(14, 12, 18);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.02; // Keep camera above floor
    controls.minDistance = 4;
    controls.maxDistance = 45;
    controls.target.set(0, 2, 0);
    controlsRef.current = controls;

    // 7. Ground Grid & Contact Shadow Plane
    const floorRadius = 32;
    const gridHelper = new THREE.GridHelper(floorRadius * 2, 64, 0x06b6d4, 0x1e293b);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);

    // Procedural contact shadow disc
    const contactCanvas = document.createElement('canvas');
    contactCanvas.width = 256;
    contactCanvas.height = 256;
    const cCtx = contactCanvas.getContext('2d')!;
    const grad = cCtx.createRadialGradient(128, 128, 20, 128, 128, 128);
    grad.addColorStop(0, 'rgba(0, 0, 0, 0.7)');
    grad.addColorStop(0.5, 'rgba(0, 0, 0, 0.25)');
    grad.addColorStop(1, 'transparent');
    cCtx.fillStyle = grad;
    cCtx.fillRect(0, 0, 256, 256);

    const contactTexture = new THREE.CanvasTexture(contactCanvas);
    const contactShadow = new THREE.Mesh(
      new THREE.PlaneGeometry(16, 16),
      new THREE.MeshBasicMaterial({
        map: contactTexture,
        transparent: true,
        depthWrite: false,
        opacity: 0.6,
      })
    );
    contactShadow.rotation.x = -Math.PI / 2;
    contactShadow.position.y = 0.01;
    scene.add(contactShadow);

    // 8. Composite 10-Part Procedural Hero Subject: "The Algorithm Observatory"
    const heroGroup = new THREE.Group();
    heroGroup.position.set(0, 0, 0);

    // Part 1: Stepped Octagonal Foundation (Primary Silhouette)
    const baseGeo = new THREE.CylinderGeometry(4.2, 4.8, 0.6, 8);
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.85,
      roughness: 0.25,
    });
    const baseMesh = new THREE.Mesh(baseGeo, baseMat);
    baseMesh.position.y = 0.3;
    baseMesh.receiveShadow = !isSoftwareGl;
    heroGroup.add(baseMesh);

    // Part 2: Polished Brass Reaction Wheel Ring (Secondary Read)
    const brassRingGeo = new THREE.TorusGeometry(3.6, 0.18, 16, 64);
    const brassMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      metalness: 0.95,
      roughness: 0.15,
    });
    const brassRing = new THREE.Mesh(brassRingGeo, brassMat);
    brassRing.rotation.x = Math.PI / 2;
    brassRing.position.y = 0.65;
    heroGroup.add(brassRing);

    // Part 3: Gyroscopic Gimbal Tilted Ring
    const gimbalGeo = new THREE.TorusGeometry(2.8, 0.12, 16, 48);
    const gimbalMat = new THREE.MeshStandardMaterial({
      color: 0x06b6d4,
      metalness: 0.9,
      roughness: 0.2,
    });
    const gimbal = new THREE.Mesh(gimbalGeo, gimbalMat);
    gimbal.position.y = 2.2;
    gimbal.rotation.x = Math.PI / 4;
    heroGroup.add(gimbal);

    // Part 4 & 5: Quad Quantum Holographic Emitter Pylons with Crystal Nodes (Accents)
    for (let p = 0; p < 4; p++) {
      const pAngle = (p / 4) * Math.PI * 2;
      const px = Math.cos(pAngle) * 3.4;
      const pz = Math.sin(pAngle) * 3.4;

      // Pylon pillar
      const pylonGeo = new THREE.CylinderGeometry(0.12, 0.18, 2.8, 8);
      const pylonMat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        metalness: 0.8,
        roughness: 0.3,
      });
      const pylon = new THREE.Mesh(pylonGeo, pylonMat);
      pylon.position.set(px, 1.4, pz);
      heroGroup.add(pylon);

      // Emitter node tip
      const tipGeo = new THREE.SphereGeometry(0.25, 16, 16);
      const tipMat = new THREE.MeshStandardMaterial({
        color: p % 2 === 0 ? 0x06b6d4 : 0xf59e0b,
        emissive: p % 2 === 0 ? 0x06b6d4 : 0xf59e0b,
        emissiveIntensity: 0.8,
        roughness: 0.1,
      });
      const tip = new THREE.Mesh(tipGeo, tipMat);
      tip.position.set(px, 2.85, pz);
      heroGroup.add(tip);
    }

    // Part 6: Floating Geodesic Hyper-Core (Levitating Central Jewel)
    const coreGeo = new THREE.IcosahedronGeometry(1.2, 1);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0x8b5cf6,
      emissive: 0x7c3aed,
      emissiveIntensity: 0.5,
      metalness: 0.7,
      roughness: 0.15,
      wireframe: false,
    });
    const coreMesh = new THREE.Mesh(coreGeo, coreMat);
    coreMesh.position.y = 2.4;
    heroGroup.add(coreMesh);

    // Part 7: Transmissive Protective Geodesic Canopy Dome
    const domeGeo = new THREE.SphereGeometry(2.4, 32, 24, 0, Math.PI * 2, 0, Math.PI * 0.5);
    const domeMat = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      transmission: isSoftwareGl ? 0 : 0.9,
      opacity: isSoftwareGl ? 0.35 : 1,
      transparent: true,
      roughness: 0.08,
      ior: 1.5,
      thickness: 1.2,
      depthWrite: false,
    });
    const dome = new THREE.Mesh(domeGeo, domeMat);
    dome.position.y = 0.6;
    heroGroup.add(dome);

    // Part 8: Concentric Floor Energy Ring
    const floorRingGeo = new THREE.RingGeometry(4.8, 5.2, 48);
    const floorRingMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.4,
    });
    const floorRing = new THREE.Mesh(floorRingGeo, floorRingMat);
    floorRing.rotation.x = -Math.PI / 2;
    floorRing.position.y = 0.02;
    heroGroup.add(floorRing);

    // Part 9 & 10: Orbiting Satellites with Synaptic Energy Rods
    const satGeo = new THREE.OctahedronGeometry(0.35);
    const satMat = new THREE.MeshStandardMaterial({
      color: 0xec4899,
      emissive: 0xec4899,
      emissiveIntensity: 0.6,
      roughness: 0.2,
    });
    const satMesh = new THREE.Mesh(satGeo, satMat);
    satMesh.position.set(4.5, 3.2, 0);
    heroGroup.add(satMesh);

    scene.add(heroGroup);
    heroGroupRef.current = heroGroup;

    // 9. Interactive 3D Decision Surface Mesh
    const planeSize = 18;
    const planeSegments = 50;
    const surfaceGeo = new THREE.PlaneGeometry(planeSize, planeSize, planeSegments, planeSegments);
    surfaceGeo.rotateX(-Math.PI / 2);

    const surfaceMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      metalness: 0.5,
      roughness: 0.3,
      side: THREE.DoubleSide,
      wireframe: false,
    });
    const surfaceMesh = new THREE.Mesh(surfaceGeo, surfaceMat);
    surfaceMesh.position.set(0, 0, 0);
    surfaceMesh.receiveShadow = !isSoftwareGl;
    scene.add(surfaceMesh);
    surfaceMeshRef.current = surfaceMesh;

    // Gradient descent optimization probe sphere
    const ballGeo = new THREE.SphereGeometry(0.4, 24, 24);
    const ballMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xf59e0b,
      emissiveIntensity: 0.8,
      roughness: 0.1,
      metalness: 0.9,
    });
    const descentBall = new THREE.Mesh(ballGeo, ballMat);
    descentBall.castShadow = !isSoftwareGl;
    scene.add(descentBall);
    descentBallRef.current = descentBall;

    // 10. InstancedMesh for Living Data Dots
    const dotCount = Math.max(dots.length, 60);
    const dotGeo = new THREE.SphereGeometry(0.24, 16, 16);
    const dotMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.2,
      metalness: 0.6,
    });
    const instancedDots = new THREE.InstancedMesh(dotGeo, dotMat, dotCount);
    instancedDots.castShadow = !isSoftwareGl;
    instancedDots.frustumCulled = false;
    scene.add(instancedDots);
    instancedDotsRef.current = instancedDots;

    // Animation & Pacing Loop
    let clock = new THREE.Clock();

    function updateSurfaceAndDots(elapsed: number) {
      const mode = modeRef.current;
      const steepness = steepnessRef.current;

      // Update Surface Vertices
      if (surfaceMeshRef.current) {
        const pos = surfaceMeshRef.current.geometry.attributes.position;
        const count = pos.count;

        for (let i = 0; i < count; i++) {
          const vx = pos.getX(i);
          const vz = pos.getZ(i);
          let vy = 0;

          if (mode === 'loss-bowl') {
            // Convex quadratic bowl + subtle non-convex ripples
            const r2 = (vx * vx + vz * vz) / 36;
            vy = (r2 * 2.5 + Math.sin(vx * 1.5 + elapsed) * Math.cos(vz * 1.5 + elapsed) * 0.4) * steepness;
          } else if (mode === 'svm-hyperplane') {
            // Tilted separating hyperplane
            vy = (vx * 0.35 + vz * 0.25 + Math.sin(elapsed * 0.5) * 0.3) * steepness;
          } else if (mode === 'cluster-space') {
            // Multi-peak Gaussian hills
            const d1 = Math.hypot(vx - 4, vz - 3);
            const d2 = Math.hypot(vx + 4, vz + 3);
            const d3 = Math.hypot(vx, vz - 4);
            vy = (Math.exp(-d1 * 0.5) * 3 + Math.exp(-d2 * 0.5) * 2.5 + Math.exp(-d3 * 0.5) * 2.8) * steepness;
          } else if (mode === 'neural-terrain') {
            // Complex non-linear sigmoid activation landscape
            const z = vx * 0.4 + vz * 0.4;
            vy = (1 / (1 + Math.exp(-z)) * 3.5 + Math.sin(vx * 0.8 + elapsed) * 0.3) * steepness;
          }

          pos.setY(i, vy);
        }
        pos.needsUpdate = true;
        surfaceMeshRef.current.geometry.computeVertexNormals();
      }

      // Update Descent Probe Ball
      if (descentBallRef.current) {
        if (mode === 'loss-bowl') {
          descentBallRef.current.visible = true;
          const theta = elapsed * 1.2;
          const radius = Math.max(0.5, 4.5 * Math.exp(-((elapsed * 0.2) % 4)));
          const bx = Math.cos(theta) * radius;
          const bz = Math.sin(theta) * radius;
          const by = ((bx * bx + bz * bz) / 36 * 2.5 + 0.4) * steepness + 0.3;
          descentBallRef.current.position.set(bx, by, bz);
        } else {
          descentBallRef.current.visible = false;
        }
      }

      // Update Instanced Data Dots
      if (instancedDotsRef.current) {
        const mesh = instancedDotsRef.current;
        const total = mesh.count;

        for (let i = 0; i < total; i++) {
          const sample = dots[i % (dots.length || 1)];
          const baseNx = sample ? sample.x : (i % 8) / 8 + 0.05;
          const baseNy = sample ? sample.y : Math.floor(i / 8) / 8 + 0.05;
          const label = sample?.label ?? (i % 2);

          // Map [0, 1] to [-7, 7]
          const dx = (baseNx - 0.5) * 14;
          const dz = (baseNy - 0.5) * 14;

          let dy = 0.5;
          if (mode === 'loss-bowl') {
            const r2 = (dx * dx + dz * dz) / 36;
            dy = (r2 * 2.5 + 0.3) * steepness + Math.sin(elapsed * 2 + i) * 0.15;
          } else if (mode === 'svm-hyperplane') {
            dy = (label === 1 ? 1.5 : -0.8) + (dx * 0.35 + dz * 0.25) * steepness;
          } else if (mode === 'cluster-space') {
            const cluster = label === 1 ? 1 : i % 3;
            const cx = cluster === 0 ? -3.5 : cluster === 1 ? 3.5 : 0;
            const cz = cluster === 0 ? -3 : cluster === 1 ? 3 : -3.5;
            dy = 1.2 + Math.sin(elapsed * 2 + i) * 0.2;
          } else {
            dy = 1.0 + Math.sin(elapsed + i) * 0.25;
          }

          _vec3.set(dx, dy, dz);
          _quat.set(0, 0, 0, 1);
          const scale = 1.0 + Math.sin(elapsed * 3 + i) * 0.1;
          _mat4.compose(_vec3, _quat, new THREE.Vector3(scale, scale, scale));
          mesh.setMatrixAt(i, _mat4);

          // Color
          if (label === 1) _color.setHex(0xf59e0b); // amber
          else if (label === 0) _color.setHex(0x06b6d4); // cyan
          else _color.setHex(0xec4899); // pink
          mesh.setColorAt(i, _color);
        }
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      }

      // Rotate Hero Architecture
      if (heroGroupRef.current) {
        gimbal.rotation.z = elapsed * 0.6;
        gimbal.rotation.y = elapsed * 0.4;
        coreMesh.rotation.y = elapsed * 0.8;
        coreMesh.rotation.x = elapsed * 0.5;
        satMesh.position.x = Math.cos(elapsed * 1.5) * 4.2;
        satMesh.position.z = Math.sin(elapsed * 1.5) * 4.2;
      }
    }

    // Paced loop implementation
    let frameId = 0;
    function afterDraw(cb: () => void) {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    }

    function tick() {
      const delta = clock.getDelta();
      const elapsed = clock.getElapsedTime() * animSpeedRef.current;

      if (autoRotateRef.current) {
        controls.autoRotate = true;
        controls.autoRotateSpeed = 1.0;
      } else {
        controls.autoRotate = false;
      }

      controls.update();
      updateSurfaceAndDots(elapsed);
      renderer.render(scene, camera);

      if (isSoftwareGl) afterDraw(tick);
      else frameId = requestAnimationFrame(tick);
    }

    tick();
    frameIdRef.current = frameId;

    // Resize listener
    const handleResize = () => {
      if (!container) return;
      const nw = container.clientWidth;
      const nh = container.clientHeight;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup on unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      if (isSoftwareGl) clearTimeout(frameIdRef.current);
      else cancelAnimationFrame(frameIdRef.current);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    };
  }, [dots]);

  // Update wireframe toggle
  useEffect(() => {
    if (surfaceMeshRef.current) {
      const mat = surfaceMeshRef.current.material as THREE.MeshStandardMaterial;
      mat.wireframe = showWireframe;
      mat.color.setHex(showWireframe ? 0x06b6d4 : 0x1e293b);
    }
  }, [showWireframe]);

  // Camera presets
  const setCameraView = (type: 'orbit' | 'top' | 'horizon' | 'closeup') => {
    const cam = cameraRef.current;
    const ctrl = controlsRef.current;
    if (!cam || !ctrl) return;

    if (type === 'orbit') {
      cam.position.set(14, 12, 18);
      ctrl.target.set(0, 2, 0);
    } else if (type === 'top') {
      cam.position.set(0, 26, 0.1);
      ctrl.target.set(0, 0, 0);
    } else if (type === 'horizon') {
      cam.position.set(16, 2.5, 4);
      ctrl.target.set(0, 2, 0);
    } else if (type === 'closeup') {
      cam.position.set(4, 3.5, 5);
      ctrl.target.set(0, 2.2, 0);
    }
    ctrl.update();
  };

  return (
    <div className="relative w-full h-[calc(100vh-61px)] overflow-hidden bg-slate-950 select-none">
      {/* 3D WebGL Canvas Container mounted in stable ref */}
      <div ref={mountRef} className="w-full h-full block cursor-grab active:cursor-grabbing" />

      {/* Top Left Title & Dataset Badge */}
      <div className="absolute top-5 left-6 z-20 flex flex-wrap items-center gap-3 pointer-events-auto">
        <div className="p-3 bg-slate-900/90 border border-slate-800 rounded-xl backdrop-blur-md shadow-xl flex items-center gap-3">
          <div className="p-2 rounded-lg bg-cyan-500/20 text-cyan-400">
            <Box className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white tracking-tight">3D Hyperspace Stage</h2>
              <span className="text-[10px] font-mono text-cyan-300 bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800/60">
                Three.js PBR
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Interactive 3D geometry, loss surfaces & multi-part procedural observatory
            </p>
          </div>
        </div>

        {customDatasetName && (
          <div className="p-2.5 bg-emerald-950/70 border border-emerald-800/80 rounded-xl flex items-center gap-2 text-xs">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-emerald-200 font-medium">Dataset: {customDatasetName}</span>
            {onOpenIngestModal && (
              <button
                onClick={onOpenIngestModal}
                className="text-[11px] text-cyan-400 hover:text-cyan-300 hover:underline font-semibold ml-1"
              >
                Change
              </button>
            )}
          </div>
        )}
      </div>

      {/* Collapsible Viewport-First Bottom Control Dock (Consolidated, <= 15% footprint) */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 w-[94vw] max-w-[860px] pointer-events-auto transition-transform duration-300 ease-out">
        {/* Collapse / Expand Toggle Tab */}
        <div className="flex justify-center -mb-px">
          <button
            onClick={() => setIsControlsCollapsed(!isControlsCollapsed)}
            className="flex items-center gap-1.5 px-4 py-1 text-xs font-semibold text-slate-300 hover:text-white bg-slate-900 border-t border-x border-slate-800 rounded-t-lg transition-colors shadow-lg"
          >
            {isControlsCollapsed ? (
              <>
                <ChevronUp className="w-3.5 h-3.5 text-cyan-400" />
                <span>Show 3D Controls</span>
              </>
            ) : (
              <>
                <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                <span>Collapse Dock (Zen View)</span>
              </>
            )}
          </button>
        </div>

        {/* Dock Panel Body */}
        {!isControlsCollapsed && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-2xl space-y-3.5">
            {/* Row 1: Mode Switcher Tabs */}
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
              <div className="flex items-center gap-1.5 overflow-x-auto text-xs">
                <span className="text-slate-400 font-medium mr-1 shrink-0">Landscape:</span>
                {[
                  { id: 'loss-bowl', label: 'Loss Bowl (GD)' },
                  { id: 'svm-hyperplane', label: 'SVM 3D Plane' },
                  { id: 'cluster-space', label: '3D Clusters' },
                  { id: 'neural-terrain', label: 'Neural Activation' },
                ].map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setSceneMode(m.id as SceneMode)}
                    className={`px-3 py-1.5 rounded-lg font-medium whitespace-nowrap transition-all ${
                      sceneMode === m.id
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>

              {/* Auto Spin Toggle */}
              <button
                onClick={() => setIsAutoRotate(!isAutoRotate)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg border transition-colors ${
                  isAutoRotate
                    ? 'bg-cyan-950/60 border-cyan-700/80 text-cyan-300'
                    : 'bg-slate-950 border-slate-800 text-slate-400'
                }`}
              >
                <Compass className="w-3.5 h-3.5" />
                <span>{isAutoRotate ? 'Auto-Orbit: ON' : 'Auto-Orbit: OFF'}</span>
              </button>
            </div>

            {/* Row 2: Sliders & Camera Presets */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs items-center">
              {/* Steepness / Curvature Slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-slate-400">
                  <span>Surface Curvature:</span>
                  <span className="font-mono tabular-nums text-white">{surfaceSteepness.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0.4"
                  max="2.5"
                  step="0.1"
                  value={surfaceSteepness}
                  onChange={(e) => setSurfaceSteepness(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 bg-slate-950 h-1.5 rounded cursor-pointer"
                />
              </div>

              {/* Animation Speed Slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-slate-400">
                  <span>Simulation Speed:</span>
                  <span className="font-mono tabular-nums text-white">{animSpeed.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="3.0"
                  step="0.2"
                  value={animSpeed}
                  onChange={(e) => setAnimSpeed(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 bg-slate-950 h-1.5 rounded cursor-pointer"
                />
              </div>

              {/* Camera Presets */}
              <div className="flex items-center justify-between sm:justify-end gap-1.5">
                <button
                  onClick={() => setShowWireframe(!showWireframe)}
                  className={`px-2.5 py-1 text-xs font-semibold rounded-lg border transition-colors ${
                    showWireframe
                      ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white'
                  }`}
                  title="Toggle wireframe topology"
                >
                  Wireframe
                </button>
                <button
                  onClick={() => setCameraView('top')}
                  className="px-2.5 py-1 text-xs bg-slate-950 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800 transition-colors"
                >
                  Top Contour
                </button>
                <button
                  onClick={() => setCameraView('orbit')}
                  className="px-2.5 py-1 text-xs bg-slate-950 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800 transition-colors"
                >
                  Reset View
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
