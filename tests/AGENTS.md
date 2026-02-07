# tests/ — Python Test Suite

Pytest-based tests for teleop_xr backend.

## OVERVIEW
All Python tests centralized here. Uses `@pytest.mark.anyio` for async WebSocket tests.

## STRUCTURE
```
tests/
├── test_*.py           # Feature tests
├── fixtures/           # Test data (URDFs, configs)
└── __init__.py
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| IK tests | `test_ik_*.py` | Solver, controller, robots |
| WebSocket tests | `test_teleop_state_sync.py` | Uses anyio |
| RAM tests | `test_ram*.py` | Resource loading |

## CONVENTIONS

### Async Testing
- Marker: `@pytest.mark.anyio`
- Fixtures: Use `anyio` fixture for event loop

### Mocking
- `unittest.mock.patch` for file system
- Mock `ram.get_resource` for offline tests

## ANTI-PATTERNS (tests)

- ❌ NEVER use real network in tests
- ✅ ALWAYS use `tmp_path` for file writes

## NOTES

**Boilerplate**: `test_ram.py` has 504 lines (high repetition, refactor candidate)
