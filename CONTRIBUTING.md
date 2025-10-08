# Contributing Guidelines

Thanks for improving the GPU Glossary! This repository contains the source for the Modal GPU Glossary. Please follow these guidelines to propose new entries or edit existing ones.

## Entry structure (required)
Each entry should use this structure:
- Definition: one-paragraph, precise and vendor-agnostic where possible
- Why it matters: how this concept impacts performance, cost, or developer ergonomics
- Key takeaways: 3–6 bullets
- Minimal example (optional): short pseudo-code or command illustrating the concept
- References: links to authoritative docs / articles

## Style
- English, concise, technically accurate, neutral tone
- Prefer generic concepts; call out vendor specifics explicitly (e.g., NVIDIA terms)
- Use SI units and consistent terminology across entries
- Avoid marketing language; cite measurable effects when possible

## Files and naming
- Place entries in `gpu-glossary/`
- Use lowercase kebab-case filenames (e.g., `memory-coalescing.md`)
- Keep SVG figures simple; place them under `gpu-glossary/assets/` and reference with relative paths

## PR checklist
- [ ] Entry follows structure and style above
- [ ] Links are valid and from reputable sources
- [ ] Figures (if any) are simple, self-contained SVG
- [ ] Spelling/grammar pass
- [ ] Commit message explains scope

## License
Content in `gpu-glossary/` is under CC BY 4.0. By contributing, you agree your changes will be licensed accordingly.
