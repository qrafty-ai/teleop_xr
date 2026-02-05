# Learnings

## TypeScript Integration with Zustand
- When extending store functionality without modifying the core store definition (to decouple tasks), locally extending types via intersection (`TeleopSettings & Partial<AdvancedSettings>`) and using type assertions (`as`) works effectively.
- This allows compilation (`npm run build`) to pass even when the underlying store hasn't been updated yet.

## Next.js Linting Issue
- Encountered `Invalid project directory provided` error when running `npm run lint` or `next lint` in the `webxr` directory.
- `npm run build` succeeds, indicating type safety and general code validity.
- The lint error appears to be environment-specific or related to how `next lint` parses arguments in this setup.

### WebXR Store Integration
- **Zustand Subscription**: The current store implementation does not use `subscribeWithSelector`. When subscribing to state changes, use the 1-argument version and manually track previous values if selector-like behavior is needed.
- **Update Rate**: `TeleopSystem` update rate can be dynamically adjusted by subscribing to `advancedSettings.updateRate` and updating `this.updateInterval = 1 / updateRate`.
- **Console Filtering**: Log filtering in `ConsoleStream` should be done before stringifying and sending payloads to minimize overhead.
- **Robot Spawning**: Robot model visibility and position can be controlled via the store. Use `robotResetTrigger` as a simple counter/timestamp to trigger re-positioning logic.
## 2026-02-05 Integration Learnings
- Zustand `persist` middleware requires lazy initialization `createJSONStorage(() => localStorage)` to be compatible with Next.js SSR.
- Subscribing to the store in class-based systems (like `TeleopSystem`) allows for reactive updates without re-initializing the whole system.
- Filtering console logs by priority on the client side reduces WebSocket traffic and terminal noise on the server.
- The `robotResetTrigger` pattern (using a timestamp to trigger an action) is a clean way to bridge React UI events to ECS-like system logic.
