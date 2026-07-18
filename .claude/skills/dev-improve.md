---
name: dev-improve
description: Use examples for many languages to improve dev setups, mostly for making working with Claude Code more efficient
---

# Location or sample programs

The examples are in /Users/markwatson/GITHUB/Claude_Code_language_integrations

 $ pwd
/Users/markwatson/GITHUB/Claude_Code_language_integrations
Marks-MacBook-Air:Claude_Code_language_integrations $ ls
golang_test	java_test README.md typescript_test

# Directions

- Determine the language used in the current directory (Python, Go golang, Java, TypeScript, or Haskell)
- Read the example files for the language used in the current directory to improve the local dev setup.

All Makefile's in sub-directories should have targets 'check' to do a fast syntax and other possible checks, 'test' to run tests, and 'clean'.

Top level directory should have a Makefile with targets 'check' 'test' 'clean' to run all sub-directory Makefile's
