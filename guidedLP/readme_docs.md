# Documentation Guide for Claude Code Integration

## Overview
This documentation structure is specifically designed to work seamlessly with Claude Code for implementing this large-scale network analysis project. The documents are organized to provide progressively more detailed specifications that Claude Code can reference during implementation.

## Document Hierarchy

```
📁 Documentation Structure
│
├── .clinerules                          ← START HERE: High-level project instructions
├── README.md                            ← Project overview for humans
│
└── 📁 docs/
    │
    ├── technical_requirements.md        ← Performance constraints & tech stack
    │
    ├── 📁 architecture/                 ← System design documents
    │   ├── overview.md                  ← Complete system architecture
    │   ├── network_construction.md      ← (To be created)
    │   ├── guided_label_propagation.md  ← Detailed GLP methodology
    │   └── timeseries_analysis.md       ← (To be created)
    │
    └── 📁 specifications/               ← Concrete function specs
        ├── network_functions.md         ← Network module functions
        ├── glp_functions.md             ← GLP module functions
        └── timeseries_functions.md      ← Time-series module functions
```

## How to Use with Claude Code

### Phase 1: Project Setup
**Goal**: Set up project structure and foundational utilities

**Claude Code Instructions**:
```bash
claude code "Read .clinerules and docs/architecture/overview.md. 
Create the project directory structure with:
- src/ with network/, glp/, timeseries/, and common/ subdirectories
- tests/ directory
- pyproject.toml with all required dependencies (networkit, polars, numpy, scipy, pytest)
- Basic README.md with project overview"
```

### Phase 2: Common Utilities
**Goal**: Build shared components used by all modules

**Claude Code Instructions**:
```bash
claude code "Read docs/specifications/network_functions.md, specifically the IDMapper class section.
Implement src/common/id_mapper.py with the IDMapper class including all specified methods.
Include comprehensive docstrings and type hints. Add unit tests in tests/test_id_mapper.py."
```

```bash
claude code "Read docs/technical_requirements.md section on input data formats.
Implement src/common/validators.py with functions to:
- Validate edge list DataFrames (required columns, data types)
- Validate timestamp formats
- Validate seed label formats
Include error handling and clear error messages."
```

### Phase 3: Network Module (Base Functionality)
**Goal**: Implement core network construction and analysis

**For each major function, use this pattern**:

```bash
claude code "Read docs/specifications/network_functions.md, specifically the 
build_graph_from_edgelist() function. Implement this function in 
src/network/construction.py following the exact specifications. 
Pay special attention to:
- Using Polars for data loading
- Preserving original node IDs with IDMapper
- Supporting both directed and undirected graphs
- Auto-weight calculation from duplicate edges
Include type hints, docstrings with complexity notation, and unit tests."
```

**Key Implementation Order**:
1. `build_graph_from_edgelist()` - Foundation for everything
2. `project_bipartite()` - Needed before other analysis
3. `extract_centrality()` - Basic analysis capability
4. `filter_graph()` - Preprocessing utility
5. `detect_communities()` - Advanced analysis
6. `apply_backbone()` - Network reduction
7. `export_graph()` - Output functionality

### Phase 4: GLP Module
**Goal**: Implement Guided Label Propagation

**Start with core propagation**:
```bash
claude code "Read docs/architecture/guided_label_propagation.md thoroughly to understand 
the conceptual framework. Then read docs/specifications/glp_functions.md, specifically 
the guided_label_propagation() function.

Implement src/glp/propagation.py with:
- guided_label_propagation() function using matrix-based calculation
- Support for both directed and undirected graphs
- Directional analysis (in-degree and out-degree) for directed graphs
- Efficient convergence checking

Use scipy.sparse for adjacency matrices. Follow the iterative propagation algorithm 
specified in the documentation. Include comprehensive tests."
```

**Then add validation capabilities**:
```bash
claude code "Read docs/specifications/glp_functions.md, specifically the validation 
functions. Implement src/glp/validation.py with:
- train_test_split_validation()
- external_validation()
- cross_validate()

Use sklearn for splits if needed. Return detailed metrics dictionaries as specified."
```

### Phase 5: Time-Series Module
**Goal**: Implement temporal network analysis

```bash
claude code "Read docs/specifications/timeseries_functions.md, specifically 
create_temporal_slices(). Implement src/timeseries/slicing.py with:
- create_temporal_slices() supporting daily/weekly/monthly/yearly intervals
- Rolling window functionality
- Cumulative vs non-cumulative options
- Consistent ID mapping across slices using align_node_ids_across_slices()

Use Polars datetime functionality for efficient date handling. Include tests with 
sample temporal data."
```

### Phase 6: Integration & Testing
**Goal**: Ensure all modules work together

```bash
claude code "Create integration tests in tests/test_integration.py that:
- Build a network from edge list
- Apply GLP with seed nodes
- Create temporal slices and track metrics over time
- Export results in multiple formats

Use the synthetic test data in tests/fixtures/ and verify end-to-end workflows."
```

## Iterative Development Pattern

### When implementing each function:

1. **Read the specification first**:
   ```bash
   claude code "Read docs/specifications/[module]_functions.md, 
   specifically the [function_name] section. Summarize the requirements."
   ```

2. **Implement with full context**:
   ```bash
   claude code "Using the specification you just read, implement [function_name] 
   in src/[module]/[file].py. Include:
   - Full function signature with type hints
   - Comprehensive docstring with parameters, returns, and complexity
   - All edge case handling specified
   - Performance optimizations mentioned in docs/technical_requirements.md"
   ```

3. **Add tests immediately**:
   ```bash
   claude code "Create unit tests for [function_name] in tests/test_[module].py.
   Test all edge cases mentioned in the specification."
   ```

4. **Benchmark if needed**:
   ```bash
   claude code "Create performance benchmark for [function_name] in benchmarks/.
   Test with the standard network sizes specified in docs/technical_requirements.md
   (1K, 10K, 100K nodes)."
   ```

## Updating Documentation

### When requirements change:

1. **Update specifications first**:
   - Modify the relevant docs/specifications/ file
   - Commit changes with clear description

2. **Inform Claude Code of changes**:
   ```bash
   claude code "Read the updated docs/specifications/[module]_functions.md, 
   specifically the [function_name] section. Update the implementation in 
   src/[module]/[file].py to match the new specification. Note the changes 
   from the previous version."
   ```

3. **Update tests**:
   ```bash
   claude code "Update tests to reflect the new specification for [function_name]."
   ```

## Common Claude Code Patterns

### Debugging Issues
```bash
claude code "The [function_name] is failing with [error]. 
Review the implementation in src/[module]/[file].py against the specification in 
docs/specifications/[module]_functions.md. Identify the discrepancy and fix it."
```

### Adding New Features
```bash
claude code "I want to add [new feature] to the [module]. 
Before implementing, review docs/architecture/[module].md and suggest:
1. Where this feature should be implemented
2. What existing functions it depends on
3. Any new specifications needed"
```

### Performance Optimization
```bash
claude code "Review [function_name] in src/[module]/[file].py. 
Using the performance requirements in docs/technical_requirements.md, 
identify optimization opportunities. Focus on:
- Using NetworkIt built-ins where possible
- Vectorization with NumPy
- Opportunities for multiprocessing"
```

### Code Review
```bash
claude code "Review the implementation of [module] against all specifications in 
docs/specifications/[module]_functions.md. Create a checklist of:
- ✅ Correctly implemented functions
- ⚠️ Functions with minor deviations
- ❌ Missing or incorrect implementations
- 💡 Suggested improvements"
```

## Best Practices

### 1. Always Reference Documentation
- Don't rely on Claude Code's memory of previous context
- Explicitly point to the relevant documentation files
- This ensures consistency across sessions

### 2. Implement Incrementally
- Build one function at a time
- Test before moving to the next
- Keep specifications updated as you learn

### 3. Use Concrete Examples
```bash
claude code "Using the sample edge list in tests/fixtures/sample_network.csv,
demonstrate how to use build_graph_from_edgelist() and extract_centrality().
Show the complete workflow with expected output."
```

### 4. Version Control Integration
- Commit after each major function implementation
- Reference Git commits in documentation updates
- Tag performance benchmarks with commits

### 5. Handle Ambiguities Early
```bash
claude code "The specification for [function_name] is ambiguous about [aspect].
Suggest 2-3 implementation options with pros/cons. I'll update the specification
based on your recommendation."
```

## Troubleshooting

### "Claude Code isn't following the specification"
✅ **Solution**: Be more explicit in your instruction
```bash
# Instead of:
claude code "Implement the GLP function"

# Use:
claude code "Read docs/specifications/glp_functions.md, specifically the 
guided_label_propagation() function section from line 8 to line 145. 
Implement this function EXACTLY as specified, paying special attention to:
- The matrix formulation in the 'Matrix Formulation' subsection
- The iterative propagation algorithm
- The directional analysis logic
Return the implementation and confirm each specification requirement is met."
```

### "Implementation doesn't match multiple runs"
✅ **Solution**: Documentation is source of truth
```bash
claude code "There's a discrepancy between the current implementation of [function]
and the specification. Please:
1. Show the current implementation
2. Show the specification requirements
3. List all differences
4. Update implementation to match specification exactly"
```

### "Need to add functionality not in specs"
✅ **Solution**: Update docs first, then implement
```bash
# Step 1: Update specification
# (Edit docs/specifications/[module]_functions.md manually or with Claude)

# Step 2: Implement
claude code "Review the newly added specification for [function] in 
docs/specifications/[module]_functions.md. Implement it following the same 
pattern as existing functions in this module."
```

## Quick Reference

| Task | Command Pattern |
|------|-----------------|
| Setup | `claude code "Read .clinerules and docs/architecture/overview.md. [task]"` |
| Implement Function | `claude code "Read docs/specifications/[module]_functions.md section [function]. Implement in src/[module]/[file].py"` |
| Add Tests | `claude code "Create tests for [function] covering all edge cases in specification"` |
| Debug | `claude code "[Error]. Review implementation against docs/specifications/[module]_functions.md"` |
| Optimize | `claude code "Optimize [function] per docs/technical_requirements.md. Focus on [aspect]"` |
| Review | `claude code "Review [module] implementation against specifications. Create checklist"` |

## Documentation Maintenance

### Keep Documentation Current
- ✅ Update specs when requirements change
- ✅ Add examples as you discover useful patterns
- ✅ Document performance characteristics after benchmarking
- ✅ Note any deviations from original plan

### Review Periodically
- After major features: Review architecture docs
- After optimization: Update performance notes
- After bug fixes: Clarify ambiguous specifications
- After user feedback: Add usage examples

---

## Summary

The key to successful Claude Code integration is:

1. **Comprehensive, structured documentation** (you have this now!)
2. **Explicit references** to documentation in every Claude Code instruction
3. **Incremental implementation** with testing at each step
4. **Documentation as source of truth** for all implementation decisions

This approach ensures:
- ✅ Consistent implementation across multiple Claude Code sessions
- ✅ Clear requirements that prevent ambiguity
- ✅ Maintainable codebase with documented design decisions
- ✅ Easy onboarding for new contributors (human or AI)

Start with Phase 1 (Project Setup) and work through systematically. The documentation structure will guide both you and Claude Code through the entire implementation process.
