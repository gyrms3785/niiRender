import { mat4 } from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.3/+esm';

const shaderCode = `
struct Uniforms {
  invAffine : mat4x4<f32>,
  invViewProj : mat4x4<f32>,
  volumeDims : vec3f,
  thresholdLow : f32,
  thresholdHigh : f32,
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
  let maxDistance = 2.0;
  let stepSize = 0.005;
  var sum = 0.0;

  for (var t = 0.0; t < maxDistance; t += stepSize) {
    let worldPos = in.rayOrigin + t * in.rayDir;
    let voxelPos4 = uniforms.invAffine * vec4f(worldPos, 1.0);
    let voxelPos = voxelPos4.xyz;

    if (all(voxelPos >= vec3f(0.0)) && all(voxelPos <= vec3f(1.0))) {
      let texCoord = voxelPos * (uniforms.volumeDims - vec3f(1.0));
      if (all(texCoord >= vec3f(0.0)) && all(texCoord <= uniforms.volumeDims - vec3f(1.0))) {
        let d = textureLoad(volume, vec3u(texCoord), 0).r;
        let weight = smoothstep(uniforms.thresholdLow, uniforms.thresholdHigh, d);
        sum += d * weight * 0.0001;
      }
    }
  }

  return vec4f(sum, sum, sum, 1.0);
}
`;

async function loadVolumeJSON(path) {
  const res = await fetch(path);
  const json = await res.json();
  const voxel = new Float32Array(json.voxel_data.flat(3));
  const dimsRaw = json.shape; // [192, 256, 256]
  const dims = [dimsRaw[1], dimsRaw[2], dimsRaw[0]]; // [x, y, z]
  const affine = new Float32Array(json.affine.flat());
  return { voxel, dims, affine };
}

async function parseVoxelTxt(url) {
  const res = await fetch(url);
  const text = await res.text();
  const lines = text.split('\n');

  let section = null;
  const shape = [];
  const affineValues = [];
  let voxelLine = "";

  for (const line of lines) {
    if (line.startsWith('--')) {
      section = line.trim();
      continue;
    }

    if (section === '--header--') {
      shape.push(...line.trim().split(' ').map(Number));
    } else if (section === '--affine--') {
      affineValues.push(...line.trim().split(' ').map(Number));
    } else if (section === '--voxel--') {
      voxelLine += line.trim() + " ";
    }
  }

  const affine = new Float32Array(affineValues);
  const voxel = new Float32Array(voxelLine.trim().split(' ').map(Number));

  const dims = [shape[1], shape[2], shape[0]]; // x, y, z 순서로

  return { voxel, dims, affine };
}

async function main(voxelData, dims, affine) {
  const canvas = document.getElementById("canvas");
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const texture = device.createTexture({
    size: dims,
    format: 'r32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    dimension: '3d',
  });

  device.queue.writeTexture(
    { texture },
    voxelData,
    { bytesPerRow: dims[0] * 4, rowsPerImage: dims[1] },
    dims
  );

  const shaderModule = device.createShaderModule({ code: shaderCode });
  const uniformBufferSize = 64 + 64 + 32;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '3d', sampleType: 'unfilterable-float' } },
    ]
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: shaderModule, entryPoint: 'vsMain' },
    fragment: { module: shaderModule, entryPoint: 'fsMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: texture.createView() },
    ],
  });

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
    const invAffine = mat4.invert(mat4.create(), affine);

    const low = parseFloat(document.getElementById('low').value);
    const high = parseFloat(document.getElementById('high').value);
    document.getElementById('lowValue').textContent = low;
    document.getElementById('highValue').textContent = high;

    const uniformData = new Float32Array(uniformBufferSize / 4);
    uniformData.set(invAffine, 0);
    uniformData.set(invViewProj, 16);
    uniformData.set(dims, 32);
    uniformData[35] = low;
    uniformData[36] = high;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData.buffer);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }]
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, 1, 0, 0);
    pass.end();
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(render);
  }

  render();
}

// loadVolumeJSON('./main/src/voxel_data.json').then(({ voxel, dims, affine }) => {
//   main(voxel, dims, affine);
// });


parseVoxelTxt('./main/src/voxel_data.txt').then(({ voxel, dims, affine }) => {
  main(voxel, dims, affine);
});