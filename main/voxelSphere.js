import { mat4 } from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.3/+esm';

// ===== 1. Volume 데이터 생성 =====
function generateSphereVolume(size) {
    const data = new Float32Array(size * size * size);
    const center = size / 2;
    const radius = size / 3;
    for (let z = 0; z < size; z++) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - center;
                const dy = y - center;
                const dz = z - center;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                const i = x + y * size + z * size * size;
                data[i] = dist < radius ? 1.0 - dist / radius : 0.0;
            }
        }
    }
    return data;
}

// ===== 2. WebGPU 초기화 =====
async function initWebGPU() {
    const canvas = document.getElementById('canvas');
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu');

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    return { device, context, format, canvas };
}

// ===== 3. Volume Texture 생성 =====
function createVolumeTexture(device, data, size) {
    const texture = device.createTexture({
        size: [size, size, size],
        format: 'r32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        dimension: '3d',
    });

    device.queue.writeTexture(
        { texture },
        data,
        { bytesPerRow: size * 4, rowsPerImage: size },
        { width: size, height: size, depthOrArrayLayers: size }
    );

    return texture;
}

// ===== 4. WGSL Shader =====
const shaderCode = `
struct Uniforms {
  invViewProj : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var volume : texture_3d<f32>;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0) rayOrigin : vec3<f32>,
  @location(1) rayDir : vec3<f32>,
};

@vertex
fn vsMain(@builtin(vertex_index) idx : u32) -> VertexOut {
  var pos = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );

  let uv = pos[idx];
  let near = vec4<f32>(uv, 0.0, 1.0);
  let far = vec4<f32>(uv, 1.0, 1.0);

  let worldNear = (uniforms.invViewProj * near).xyz / (uniforms.invViewProj * near).w;
  let worldFar = (uniforms.invViewProj * far).xyz / (uniforms.invViewProj * far).w;

  var out : VertexOut;
  out.position = vec4<f32>(uv, 0.0, 1.0);
  out.rayOrigin = worldNear;
  out.rayDir = normalize(worldFar - worldNear);
  return out;
}

@fragment
fn fsMain(in: VertexOut) -> @location(0) vec4<f32> {
  let step = 0.01;
  var sum = 0.0;
  for (var t = 0.0; t < 2.0; t += step) {
    let p = in.rayOrigin + t * in.rayDir;
    if (all(p >= vec3<f32>(0.0)) && all(p <= vec3<f32>(1.0))) {
      let d = textureLoad(volume, vec3<u32>(p * 64.0), 0).r;
      sum += d * 0.05;
    }
  }
  return vec4<f32>(sum, sum, sum, 1.0);
}
`;

// ===== 5. Main 함수 =====
async function main() {
    const size = 64;
    const volumeData = generateSphereVolume(size);
    const { device, context, format, canvas } = await initWebGPU();
    const volumeTexture = createVolumeTexture(device, volumeData, size);

    const shaderModule = device.createShaderModule({ code: shaderCode });

    const uniformBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {
                    viewDimension: '3d',
                    sampleType: 'unfilterable-float',
                }
            },
        ],
    });

    const pipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        vertex: { module: shaderModule, entryPoint: 'vsMain' },
        fragment: {
            module: shaderModule,
            entryPoint: 'fsMain',
            targets: [{ format }],
        },
        primitive: { topology: 'triangle-list' },
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: volumeTexture.createView() },
        ],
    });

    // ===== 카메라 변수 =====
    let cameraTheta = Math.PI / 2;
    let cameraPhi = Math.PI / 2;
    let cameraRadius = 2.5;
    let cameraTarget = [0.5, 0.5, 0.5];
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener('mousedown', (e) => {
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
    });
    canvas.addEventListener('mouseup', () => dragging = false);
    canvas.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
        cameraTheta -= dx * 0.005;
        cameraPhi -= dy * 0.005;
        cameraPhi = Math.max(0.05, Math.min(Math.PI - 0.05, cameraPhi));
    });
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        cameraRadius *= 1 + e.deltaY * 0.001;
        cameraRadius = Math.max(0.5, Math.min(10.0, cameraRadius));
    }, { passive: false });

    function render() {
        const aspect = canvas.width / canvas.height;
        const proj = mat4.perspective(mat4.create(), Math.PI / 4, aspect, 0.1, 100);

        const cx = cameraTarget[0] + cameraRadius * Math.sin(cameraPhi) * Math.cos(cameraTheta);
        const cy = cameraTarget[1] + cameraRadius * Math.cos(cameraPhi);
        const cz = cameraTarget[2] + cameraRadius * Math.sin(cameraPhi) * Math.sin(cameraTheta);
        const eye = [cx, cy, cz];

        const view = mat4.lookAt(mat4.create(), eye, cameraTarget, [0, 1, 0]);
        const viewProj = mat4.multiply(mat4.create(), proj, view);
        const invViewProj = mat4.invert(mat4.create(), viewProj);

        device.queue.writeBuffer(uniformBuffer, 0, invViewProj);

        const commandEncoder = device.createCommandEncoder();
        const pass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }],
        });

        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(6, 1, 0, 0);
        pass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(render);
    }

    render();
}

main();
