
## Robot Name Property
- Decision: Added `name` as an abstract property in `BaseRobot`.
- Rationale: Needed a reliable way to identify robots for asset loading (like sphere decomposition JSONs) without hardcoding paths in each subclass. Subclasses will implement this to return their specific identifier (e.g., "h1", "g1").
