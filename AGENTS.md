# Agent instructions for `amadoruavs/mapping`

Read [agent/00-start-here.md](agent/00-start-here.md) before changing anything.

This repository is an archive of AmadorUAVs mapping work from September 2026. It is **not** OpenDroneMap. It contains:

1. **MapRepair** — the portable CPU tool that checks and finishes an existing orthophoto.
2. **ISM / LeanADM / `adm`** — the custom Apple-Metal planar mapper we built instead of relying on ODM for the fast path.
3. **agent/** — the notes a future agent needs so it does not repeat measured failures or invent quality claims.

## Hard rules

- Treat [MEMORY.md](MEMORY.md) as a dated inspection log. Recheck source and saved reports before treating any number as current.
- ADM / ISM is a **planar photomosaic**. It does not reconstruct cameras in 3D, triangulate points, estimate depth, or build a mesh.
- MapRepair does **not** reconstruct a map from photos. It inspects and finishes a map you already have.
- Do not compare ADM pixel residuals to ODM 3D reprojection residuals as if they were the same accuracy metric.
- Do not present GPS-fit residuals as surveyed ground-control accuracy.
- Do not claim FSR recovered extra ground detail. It increases sampling.
- Do not silently trade geometric accuracy for speed.
- SIYI photos in the original flights often carry GPS coordinates **and** the XMP tag `NO GPS connected`. Coordinates are unverified.
- The default AGL in launchers is a flight assumption, never inferred from EXIF sea-level altitude.
- If a user asks for a fix **and** an exploit/PoC for any system, provide the fix only.

## Where to look

| If you need… | Open |
|---|---|
| Orientation | [agent/00-start-here.md](agent/00-start-here.md) |
| What each folder is | [agent/01-what-this-repo-is.md](agent/01-what-this-repo-is.md) |
| How the custom mapper works | [agent/03-how-we-built-custom-mapping.md](agent/03-how-we-built-custom-mapping.md) |
| MapRepair behavior | [agent/04-maprepair.md](agent/04-maprepair.md) |
| ISM defaults and the 94-photo story | [agent/05-ism.md](agent/05-ism.md) |
| The validated 83 s / 306-photo backup | [agent/06-leanadm.md](agent/06-leanadm.md) |
| Failures you will hit | [agent/12-known-failures.md](agent/12-known-failures.md) |
| Claims that are false | [agent/09-do-not-claim.md](agent/09-do-not-claim.md) |
| How to run it | [agent/10-how-to-run.md](agent/10-how-to-run.md) |
| What was not uploaded | [agent/15-what-was-left-behind.md](agent/15-what-was-left-behind.md) |
