#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { WASI } from "node:wasi";

if (process.argv.length !== 6) {
  console.error("usage: run_wasi_trace.mjs trace.wasm model.bin prefix forced");
  process.exit(2);
}

const [, , wasmPath, modelPath, prefix, forced] = process.argv;
const wasi = new WASI({
  version: "preview1",
  args: ["runtime_trace", `/model/${path.basename(modelPath)}`, prefix, forced],
  env: {},
  preopens: { "/model": path.dirname(path.resolve(modelPath)) },
});
const module = await WebAssembly.compile(fs.readFileSync(wasmPath));
const instance = await WebAssembly.instantiate(module, {
  wasi_snapshot_preview1: wasi.wasiImport,
});
wasi.start(instance);
