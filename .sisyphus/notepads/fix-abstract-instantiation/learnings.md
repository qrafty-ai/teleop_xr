## Abstract Class Instantiation in Tests
When adding new abstract methods or properties to a base class, ensure all mock implementations in tests are updated to include them, otherwise instantiation will fail with TypeError.
