> Historical record (superseded by the conventional design).

# Air2stream pinned-source build and reference-case audit

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Verdict | **BLOCKED_NO_COMPILER / REFERENCE_CASE_NOT_ATTESTABLE** |
| Upstream | `https://github.com/spiccolroaz/air2stream` |
| Pinned commit | `d4834bccf01657c03ab60efb4c18f8a256132c53` |
| Tree | `daa0ecffa6125c3b50cac4e43581d60da68164e7` |
| Scope | Isolated, low-priority source/build/reference inspection in `/tmp`; no ThermoRoute data, outputs or Stage-09 resource was accessed |

This is a failed-closed build audit, not an Air2stream reproduction receipt and
not a comparator freeze. It records why the official reference case cannot yet
be attested in the current environment.

## 1. Source identity

A depth-one clone made under `nice -n 19` and `ionice -c3` resolved local `HEAD`,
remote `HEAD` and `refs/heads/master` to the pinned commit. `git fsck
--no-dangling` passed and the temporary worktree was clean. No tag was advertised;
the README's “Version 1.0.0 - October 2015” is therefore descriptive text rather
than a signed/tagged release identity. The commit is not cryptographically
signed (`%G? = N`; `git verify-commit` fails), so TLS transport plus Git object
identity is the available provenance, not signer attestation.

The temporary checkout remains at
`/tmp/air2stream-verify.KCI9VW/repo` and is approximately 32 MiB. It is not a
project artifact and may be discarded independently after this audit.

## 2. Build blocker

No supported Fortran compiler is installed: `gfortran`, `ifort`, `ifx`, `flang`,
`nvfortran`, `f95` and `f90` were all absent. The repository supplies no Makefile
or documented compiler command. `AIR2STREAM_READ.f90` imports Intel-specific
`ifport` and calls `makedirqq`, so a portable gfortran build would in any event
require either an upstream-supported route or an explicitly reviewed source
port. Installing a compiler or changing upstream source was outside this audit.

The shipped `air2stream_1.0.0.out` Linux binary was not executed: an upstream
binary run would not prove that the pinned source builds, and the repository does
not publish a golden-output checksum against which to attest it.

## 3. Reference-case limitations

The repository contains three Switzerland calibration/validation pairs:
`MAH_2369`, `SIO_2011` and `DAV_2327`. The default `input.txt` selects SIO 2011,
calibration mode, daily support, eight parameters, RMS objective, CRN mode, PSO
and 500 iterations. The default calibration and validation inputs contain 7,671
and 3,287 rows respectively; all six case files contain 18,629 rows.

There is no committed expected output. The default PSO path repeatedly calls
Fortran `random_seed`/`random_number` without a fixed seed contract, so an
eventual reference must use tolerance-based declared metrics or an explicitly
frozen deterministic seed path rather than demand byte-identical optimizer
output. The README attributes preprocessed inputs to FOEN and MeteoSwiss; the
repository's CC BY-SA 3.0 code licence does not by itself resolve upstream case-
data redistribution.

## 4. Verified byte identities

| Artifact | SHA-256 |
| --- | --- |
| `LICENSE` | `7bbaf1916c0b9856d478b30ca71eae7b3f15c566c48dbdb1b9e2172588c21d95` |
| `src/AIR2STREAM_MAIN.f90` | `aca255498623ad2fe6f7ee7dc2a2bd42447ce9dcef5d92e9152725938b3a6be6` |
| `src/AIR2STREAM_MODULES.f90` | `74fae2a7367b111c4dfc7c8a82291e37d7503cc25e1e0847e433ab05b341a73e` |
| `src/AIR2STREAM_READ.f90` | `95c1691bdac71fe527607b375b09ac9f49a8eff972278101acd81d556e4178c6` |
| `src/AIR2STREAM_RUNMODE.f90` | `e231586019b8b8ba3c00c43269fd308bb71baf6abff72425b6e4d8e27ee72de8` |
| `src/AIR2STREAM_SUBROUTINES.f90` | `0b3c3890695aa124cf8fe0448718ee4ba1f7069f1157b155bf11cd6b49c2ee31` |
| `input.txt` | `f05f9a346636a5b41448b1098910819b042044755b3c570416b88e1c9cbc39c6` |
| `PSO.txt` | `f9f312ffd7736da19f929a571554aa412e6bba8f6b3b8d6c0988a3a34ff6b937` |
| `Switzerland/SIO_2011_cc.txt` | `7c744894234b09335aade8d76b76c0fa86b3e4fa566dc429a2bbbcf10c405495` |
| `Switzerland/SIO_2011_cv.txt` | `c65c83f2ced6c4998ad542bef02e142e20bcfb882b15c4f7f0d5d6abd6617f8b` |
| `Switzerland/MAH_2369_cc.txt` | `038943e8c16a5c8a1a767fed3719a63116dfb06a783b835f8b75aaa9680938c7` |
| `Switzerland/MAH_2369_cv.txt` | `f31d4b360693494fe75e9b3788e2ccc7ce70a7b163d84d0a371effb26c497ae5` |
| `Switzerland/DAV_2327_cc.txt` | `96aced529be5bdb33ea9ceb7c4cb2972ee2907cfd508dc0d705d55f02ef3407f` |
| `Switzerland/DAV_2327_cv.txt` | `c8485235c40b2ec79c26ebf7b84c8f7586452af0d320a0f59f328f1d892562eb` |

The two SIO hashes and all five source hashes were independently rechecked from
the temporary checkout after the delegated audit. No output hash exists because
no model execution occurred.

## 5. Conditions for a future PASS

1. Freeze an authorized compiler/image and exact flags, or obtain an upstream-
   supported portable source revision without silently changing the comparator.
2. Resolve the reference-input redistribution and provenance scope.
3. Freeze the calibration parameterization, objective, seed policy and common
   compute/trial budget before Route-B outcomes.
4. Define a golden reference or tolerance-based metrics and expected failure
   behavior; do not invent expected output after running.
5. Record source, compiler, inputs, outputs, runtime/resource use and a self-
   hashed receipt from an isolated build.
6. Only then mark official Air2stream `BUILD_PASS` for the known-gauge arm; it
   remains ineligible as the unmodified strict-ungauged baseline.
