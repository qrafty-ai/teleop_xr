export interface GeneralConfig {
  input_mode: string;
  natural_phone_position: [number, number, number];
  natural_phone_orientation_euler: [number, number, number];
}

const STORAGE_KEY = 'teleop_general_config';
const DEFAULT_CONFIG: GeneralConfig = {
  input_mode: 'controller',
  natural_phone_position: [0, 0, 0],
  natural_phone_orientation_euler: [0, 0, 0],
};

let currentConfig: GeneralConfig = loadConfig();
const handlers: ((config: GeneralConfig) => void)[] = [];

function loadConfig(): GeneralConfig {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) {
    try {
      return { ...DEFAULT_CONFIG, ...JSON.parse(stored) };
    } catch (e) {
      console.error('Failed to parse general config', e);
    }
  }
  return { ...DEFAULT_CONFIG };
}

function saveConfig(config: GeneralConfig): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

export function getGeneralConfig(): GeneralConfig {
  return { ...currentConfig };
}

export function setGeneralConfig(config: Partial<GeneralConfig>): void {
  currentConfig = { ...currentConfig, ...config };
  saveConfig(currentConfig);
  notify();
}

export function onGeneralConfigChanged(handler: (config: GeneralConfig) => void): () => void {
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
