# Jupyter Notebook Rules

## ⛔ DO NOT

1. **Rewrite notebooks from scratch**
   - Do not delete and recreate existing notebooks
   - Do not remove all cells and rewrite them
   - Do not use `Overwrite: true` to completely replace notebooks

2. **Make unnecessary changes**
   - Do not modify working code unnecessarily
   - Do not delete existing outputs (unless required)
   - Do not reformat markdown cells without reason

3. **Break cell order**
   - Do not add/remove cells that break logical flow
   - Do not reorder cells with dependencies incorrectly

## ✅ DO

1. **Apply minimal changes**
   - Only edit the necessary cells
   - Preserve existing structure
   - Make targeted modifications

2. **Use cell-level editing**
   - Target only the specific cell that needs changes
   - Add new cells at the correct position
   - Minimize deletions

3. **Preserve outputs**
   - Keep important outputs intact
   - Do not delete visualizations (unless updating them)

## ⚠️ Exceptions

You may rewrite a notebook from scratch ONLY if:
- User explicitly says "rewrite from scratch" or "create new"
- Notebook is completely broken/unusable
- User requests a major restructure
