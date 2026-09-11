# Weekly Deploy Process

Every Tuesday, the release engineer cuts a branch from main, runs the full
test suite, and if it's green, deploys to staging. QA has until Wednesday
noon to sign off. If sign-off happens, the same branch goes to production
Wednesday afternoon. If QA finds a blocker, the branch is patched and the
test suite reruns before a second staging deploy.

Rollbacks are done by redeploying the previous production tag, not by
reverting commits on main, so main always reflects intended history.
