import { initConsoleStream } from "./console_stream.js";

import "./systems/teleop.js";
import "./components/robot-model.js";
import "./components/video-stream.js";
import "./components/ui-components.js";

initConsoleStream();

console.log("[Index] A-Frame scene entry point initialized");
