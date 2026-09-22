// Registry of live demos. To add one: write a module like flappy.js and list it here.
//
// A demo module default-exports:
//   id, title, keys, blurb      strings for the UI
//   checkpoint                  the Laya checkpoint that handles this demo's phrasing best
//   params                      [{ id, label, min, max, step, value }] sliders passed to act()
//   input(keys, pressed)        map held keys / keys pressed this frame to the demo's own input object
//   touch                       true when input() also reads pressed.has('pointer'), so a phone can play it
//   create()                    returns an instance with:
//       score, best, crashes, dead            numbers read by the HUD (dead > 0 while crashed)
//       update(dt, input)                     advance the simulation; never waits for the model
//       observe()                             -> { state, questions } sent to /api/predict
//       act(answers, params, apply)           -> { label, why }; only change the game when `apply`
//       draw(ctx, w, h)                       paint the scene in GRAYSCALE; the stage dithers it
//
// Write the state in words and ask what the model SEES, not what to do. See the Notes section.
import flappy from './flappy.js';
import runner from './runner.js';
import tetris from './tetris.js';
import snake from './snake.js';

export default [flappy, runner, tetris, snake];
