# S57 — Cleanup + fresh-clone test
**Phase:** F · **Target date:** Jan 12 · **Depends on:** S56

## Goal
Someone else can build and run this.

## Steps
1. Delete dead code and old experiments (they live in git history).
2. `infer --help`; consistent error messages; non-zero exit codes on errors.
3. Write a README **quickstart** section: prerequisites, model download, build, test, run.
4. `git clone` into `/tmp/x` and follow **only** the quickstart.

## Done when
- [ ] The fresh clone builds, passes tests, and chats, following only the README.

## Notes
