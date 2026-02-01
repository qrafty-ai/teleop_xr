import "aframe";
import { initConsoleStream } from "./console_stream.js";

import "./systems/teleop.js";
import "./components/robot-model.js";
import "./components/spatial-panel.js";
import "./components/video-stream.js";
import "./components/ui-components.js";

import "./robot_settings_system.js";
import "./camera_settings_system.js";
import "./general_settings_system.js";

initConsoleStream();

console.log("[Index] A-Frame scene entry point initialized");
