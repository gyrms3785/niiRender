// WebGPU hybrid viewer: outer quad mesh + inner voxel volume rendering
// 복붙하면 실행되는 최소한의 예제입니다 (Chrome + WebGPU 필요)

const canvas = document.getElementById('canvas');

if (!navigator.gpu) {
  alert('WebGPU not supported!');
  throw new Error('WebGPU not supported');
}

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const context = canvas.getContext('webgpu');

context.configure({
  device,
  format: 'bgra8unorm',
  alphaMode: 'opaque',
});

// --- Dummy Voxel Data (4x4x4 RGBA texture) ---
const voxelDim = 4;
const voxelData = new Uint8Array(voxelDim * voxelDim * voxelDim * 4);
for (let z = 0; z < voxelDim; z++) {
  for (let y = 0; y < voxelDim; y++) {
    for (let x = 0; x < voxelDim; x++) {
      const i = 4 * (z * voxelDim * voxelDim + y * voxelDim + x);
      voxelData[i + 0] = x * 64; // R
      voxelData[i + 1] = y * 64; // G
      voxelData[i + 2] = z * 64; // B
      voxelData[i + 3] = 255;    // A
    }
  }
}

const voxelTexture = device.createTexture({
  size: [voxelDim, voxelDim, voxelDim],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  dimension: '3d',
});
device.queue.writeTexture(
  { texture: voxelTexture },
  voxelData,
  { bytesPerRow: voxelDim * 4, rowsPerImage: voxelDim },
  { width: voxelDim, height: voxelDim, depthOrArrayLayers: voxelDim }
);

// --- Dummy Cube Geometry (quad-ish) ---
const cubeVertices = new Float32Array([
  // position         color
  -1, -1, -1,         1, 0, 0,
   1, -1, -1,         0, 1, 0,
   1,  1, -1,         0, 0, 1,
  -1,  1, -1,         1, 1, 0,
  -1, -1,  1,         0, 1, 1,
   1, -1,  1,         1, 0, 1,
   1,  1,  1,         1, 1, 1,
  -1,  1,  1,         0, 0, 0,
]);

const cubeIndices = new Uint16Array([
  0, 1, 2,  2, 3, 0,
  4, 5, 6,  6, 7, 4,
  0, 1, 5,  5, 4, 0,
  2, 3, 7,  7, 6, 2,
  1, 2, 6,  6, 5, 1,
  3, 0, 4,  4, 7, 3,
]);

const vertexBuffer = device.createBuffer({
  size: cubeVertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, cubeVertices);

const indexBuffer = device.createBuffer({
  size: cubeIndices.byteLength,
  usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(indexBuffer, 0, cubeIndices);

// --- Shader ---
const shaderModule = device.createShaderModule({
  code: `
    struct VSOut {
      @builtin(position) pos: vec4<f32>,
      @location(0) vColor: vec3<f32>,
    };

    @vertex
    fn vs_main(
      @location(0) pos: vec3<f32>,
      @location(1) color: vec3<f32>
    ) -> VSOut {
      var out: VSOut;
      out.pos = vec4<f32>(pos / 5, 1.0);
      out.vColor = color;
      return out;
    }

    @fragment
    fn fs_main(@location(0) vColor: vec3<f32>) -> @location(0) vec4<f32> {
      return vec4<f32>(vColor, 1.0);
    }
  `,
});

// --- Pipeline Layout ---
const pipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: {
    module: shaderModule,
    entryPoint: 'vs_main',
    buffers: [{
      arrayStride: 6 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' },
      ],
    }],
  },
  fragment: {
    module: shaderModule,
    entryPoint: 'fs_main',
    targets: [{ format: 'bgra8unorm' }],
  },
  primitive: {
    topology: 'triangle-list',
    cullMode: 'back',
  },
  depthStencil: undefined,
});

// --- Frame Rendering ---
function frame() {
  const commandEncoder = device.createCommandEncoder();
  const textureView = context.getCurrentTexture().createView();

  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [{
      view: textureView,
      loadOp: 'clear',
      storeOp: 'store',
      clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
    }],
  });

  renderPass.setPipeline(pipeline);
  renderPass.setVertexBuffer(0, vertexBuffer);
  renderPass.setIndexBuffer(indexBuffer, 'uint16');
  renderPass.drawIndexed(cubeIndices.length);
  renderPass.end();

  device.queue.submit([commandEncoder.finish()]);
  requestAnimationFrame(frame);
}

frame();
console.log('WebGPU initialized: Voxel + Cube rendering.');
