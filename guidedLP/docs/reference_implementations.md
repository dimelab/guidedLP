# Reference Implementations

## Overview
This document lists existing code files that should inspire the implementation of this project. Claude Code should review these files to understand the thoughts behind previously conceived functions and details regarding their implementation. It should only serve as an inspiration as the actual implementation of the functions can be done more elegantly.

## Global References (Apply to All Modules)

### File: `guidedLP_OLD/network/net_utils.py`
**Learn from this file:**
- The `stlp` function (label propagation reference)
- Matrix calculation patterns
- Various other network functions

**Key function to reference:** `stlp()`
**Adapt for our project:**
- Use our IDMapper for node ID handling
- Add directional propagation support
