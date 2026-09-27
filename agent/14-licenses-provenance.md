# Licenses and provenance

## Original work

AmadorUAVs mapping code from September 2026. Club use. No single SPDX identifier was applied to the original tree.

## Pins that shipped

| Piece | Pin / hash | Notes |
|---|---|---|
| OpenRT | git `ef6205f8ca51fd123da4a2f0b7b7a6ad27809b80` | `MapRepair/third_party/OpenRT/`; GCC math-namespace patch |
| FSR frame interpolation | FidelityFX SDK 1.1.4, rev `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55` | GLSL translated to MSL; unmodified GLSL |
| FSR1 EASU/RCAS | AMD FidelityFX | MIT; `MapRepair/FSR_LICENSE.txt` |
| OpenSfM residuals | SuperBuild rev `c5328439465e6ace011f39077d1077d7b1cdd65d` | Equations ported; AGPL upstream |
| ODM comparison clone | `1a7134170e6100cade3236486a279d6241f11f99` | Not vendored here |
| Pixi (ODM port) | 0.77.1, checksum-pinned | Not vendored here |
| OpenCV (MapRepair Mac) | Homebrew opencv@4 4.14.0 for C++; Python cv2 5.x separate |
| OpenCV (Linux container) | C++ 4.6, Python 5.0.0.93 |
| rasterio | 1.5.1 in the original venv |
| ISM.metal (rollback target) | SHA-256 `c610ad5f…e0cc2c2` | `runtime.zip` |
| Later ISM.metal (92 kernels / cinematic) | SHA-256 `7f4ed7b3…2937ff` | GUIDE era |
| MapRepair-linux.tar.gz | SHA-256 in `MapRepair-linux.tar.gz.sha256` | `94414ef2ac85f04d25835e6185d5f814b77ca17c3b3e3c7e31aa8e98b892ab0a` as of 2026-09-17 |
| XDNA wheel mentioned in notes | 1.4.4.dev3+gf50bef7 | Not installed here |
| Peano | 22.0.0.2026090701+3e93bf7b | Notes only |

`MapRepair/source_provenance.json` hashes a few ADM presentation files as they existed when MapRepair was split off. Those hashes will not match if you edit `adm/` later.

`ISM/runtime/bundle_manifest.json` lists Metal entry points (92).

## AGPL caution

`docs/OPENSFM_PORT.md` describes a residual/equation port. Shipping a fuller OpenSfM or ODM tree inside this repo would put AGPL on the combined work. Do not copy `third_party/opensfm_reference` back in casually. That tree was left behind (~53 MB).

## What we did to OpenRT

Three files: `BxDF.h` includes `<cmath>` and uses `std::sqrt`; `BRDFPhong.cpp` and `BRDFBlinn.cpp` use `std::pow`. Lighting numerically equivalent. Documented in MapRepair provenance. Do not revert "to match upstream" on GCC.

## FSR translation chain

Unmodified GLSL compiled with glslang, optimized with spirv-opt, translated with SPIRV-Cross to MSL 4.0, compiled at runtime. Requires macOS 26 coherent textures. Optical-flow assistance disabled. Motion comes from ADM homographies and a constant flat depth assumption (60 m in the 306-photo FSR experiment).
