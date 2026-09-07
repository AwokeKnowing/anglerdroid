# Skill: Door checkered mat avoid (brown border)

## Description
Avoid the black/white checkered floor by the front door. Primary cue is the
**dark brown border/transition strip** around that mat on the tan carpet —
not `findChessboardCorners` on the 30Hz path.

## Layers
1. **Named keepout** `checkered_door` — map/ego paint (preferred persistent avoid)
2. **~3Hz extras loop** — HSV brown-border detect on RS1 topdown RGB (sticky hazard)
3. **Neural RL** — learn high stuck risk near this texture and how to get unstuck
4. Depth soft-low / near-field still protect on 30Hz regardless

## Do not
- Run chessboard corner detect inside capture / 30Hz loop
- Rely only on RGB hazard without keepout + RL

## Success
- Does not drive onto the door checkered area
- 30Hz stays free for fresh frames + depth safety
