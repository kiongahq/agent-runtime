# Changelog

## [0.2.0](https://github.com/kiongahq/agent-runtime/compare/v0.1.0...v0.2.0) (2026-11-10)


### ⚠ BREAKING CHANGES

* environment variables, headers and resource names are renamed.

### Features

* **designer:** add POST /agent-turns for Agent Teams ([#48](https://github.com/kiongahq/agent-runtime/issues/48)) ([55d4941](https://github.com/kiongahq/agent-runtime/commit/55d4941ff811cdce47ebd01f0f08ddb468ee1976))
* **designer:** LLM rubric judge and nested versions in runner requests ([bc86779](https://github.com/kiongahq/agent-runtime/commit/bc86779d5ec2d9a314d41af165268f232b929626))
* **designer:** read the runner token from a mounted secret file ([2420842](https://github.com/kiongahq/agent-runtime/commit/24208425597fdbbe5462c029069e5e6b97d2d1d1))
* execute published Langflow flows through agent runtime adapter ([c233707](https://github.com/kiongahq/agent-runtime/commit/c23370778e489064a7d1a2534ddd87f100f118c4))
* **native-graph:** deployed versions stream tokens and run embedded subflows ([d637179](https://github.com/kiongahq/agent-runtime/commit/d6371796a432b023e3020988fc2c87e9a93ded9c))
* **native-graph:** error-port fallbacks and the extract component ([bdc3648](https://github.com/kiongahq/agent-runtime/commit/bdc364847e2b1ce57cde88d01b61b9ed48c89ea4))
* **native-graph:** execute Agent Designer graphs in an isolated runner ([34b1583](https://github.com/kiongahq/agent-runtime/commit/34b1583d685e94f85725d0a897e0a90ad291a40d))
* **native-graph:** parallel scheduler, router, join, flows, approval, token deltas ([ec368a0](https://github.com/kiongahq/agent-runtime/commit/ec368a02d0c8cd2a719ce650e2ac2d5028947830))
* package LFX runtime with Kionga resource components ([190b17d](https://github.com/kiongahq/agent-runtime/commit/190b17dfbb2f3f0fb8eb72d106bdd2d1c6224baf))
* **release:** add reviewed versions and consistent publication ([eed6030](https://github.com/kiongahq/agent-runtime/commit/eed6030ed47b2c173e901f3c1ab56e77c8e5ad4d))
* **release:** standardize versions, changelogs and publication ([48cbe11](https://github.com/kiongahq/agent-runtime/commit/48cbe11ab230b7116ecdd259e1f87d2d78b3fb6f))
* rename mlaiops to kionga ([#51](https://github.com/kiongahq/agent-runtime/issues/51)) ([b504f75](https://github.com/kiongahq/agent-runtime/commit/b504f758513b9b1ac4c4c2d2dd07ed953dfa1fb7))


### Fixes

* **ci:** install jsonschema for native graph contract tests ([2a3d1df](https://github.com/kiongahq/agent-runtime/commit/2a3d1dfcce0e84c390ba6be2042b64939aadb9e9))
* document fixed loopback URL in Langflow runtime component ([244c60d](https://github.com/kiongahq/agent-runtime/commit/244c60d9ebfa15490f9e3653b794d03dcb5999fb))
* **features:** wait for materializer dependencies ([56bb2a7](https://github.com/kiongahq/agent-runtime/commit/56bb2a70a08d9f35930907d5cb3435c5ed2fe9fe))
* P1 runtime security and correctness fixes ([#7](https://github.com/kiongahq/agent-runtime/issues/7) [#8](https://github.com/kiongahq/agent-runtime/issues/8) [#9](https://github.com/kiongahq/agent-runtime/issues/9) [#10](https://github.com/kiongahq/agent-runtime/issues/10) [#25](https://github.com/kiongahq/agent-runtime/issues/25) [#27](https://github.com/kiongahq/agent-runtime/issues/27) [#31](https://github.com/kiongahq/agent-runtime/issues/31)) ([#46](https://github.com/kiongahq/agent-runtime/issues/46)) ([b3a996f](https://github.com/kiongahq/agent-runtime/commit/b3a996f726539b8551145abec0371ce20e985bce))


### Documentation

* product README, CONTRIBUTING and ARCHITECTURE ([fa336a6](https://github.com/kiongahq/agent-runtime/commit/fa336a6add7937327991a037b2207f187130ce82))
* **release:** align contributor publication instructions ([7f905fb](https://github.com/kiongahq/agent-runtime/commit/7f905fb09fc4a94159f8c2c4fd84ccb1f0fdcb4a))
* **security:** clarify sole-maintainer release review ([59830c6](https://github.com/kiongahq/agent-runtime/commit/59830c65842f88e6bd3f4676974cbeeeb56aad7f))


### Build and dependencies

* pin focused LFX provider set for managed agents ([f0e4fb3](https://github.com/kiongahq/agent-runtime/commit/f0e4fb39a7b8ef3bd521efa8ed88e641d4a3225a))


### Maintenance

* ignore local tooling and secret files ([c2d52b9](https://github.com/kiongahq/agent-runtime/commit/c2d52b912b3cf71bfc1d1141f0713b6b3415ecb8))
* keep the bandit reason off the nosec line ([3fb79a7](https://github.com/kiongahq/agent-runtime/commit/3fb79a7309b534f261430be6d400cdfb01214a26))
* license repository under MIT ([473b01d](https://github.com/kiongahq/agent-runtime/commit/473b01d319688f2bb7823ebf90c2cda8277ba37e))
* **security:** record bandit review of the container listener ([97089fe](https://github.com/kiongahq/agent-runtime/commit/97089fe95a651301cc7181dba6fd0b0e531a4030))

## Changelog

Reviewed release pull requests add versioned changes here.
Existing tags and releases remain available in [GitHub](https://github.com/kiongahq/agent-runtime/releases).
