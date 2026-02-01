export interface RobotConfig {
  urdfPath: string;
  meshPath: string;
}

const STORAGE_KEY = 'teleop_robot_config';
const DEFAULT_CONFIG: RobotConfig = {
  urdfPath: 'robot.urdf',
  meshPath: '',
};

let currentConfig: RobotConfig = loadConfig();
const handlers: ((config: RobotConfig) => void)[] = [];

function loadConfig(): RobotConfig {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) {
    try {
      return { ...DEFAULT_CONFIG, ...JSON.parse(stored) };
    } catch (e) {
      console.error('Failed to parse robot config', e);
    }
  }
  return { ...DEFAULT_CONFIG };
}

function saveConfig(config: RobotConfig): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

export function getRobotConfig(): RobotConfig {
  return { ...currentConfig };
}

export function setRobotConfig(config: Partial<RobotConfig>): void {
  currentConfig = { ...currentConfig, ...config };
  saveConfig(currentConfig);
  notify();
}

export function onRobotConfigChanged(handler: (config: RobotConfig) => void): () => void {
  handlers.push(handler);
  handler({ ...currentConfig });
  return () => {
    const index = handlers.indexOf(handler);
    if (index !== -1) {
      handlers.splice(index, 1);
    }
  };
}

function notify(): void {
  const config = { ...currentConfig };
  handlers.forEach(h => h(config));
}
