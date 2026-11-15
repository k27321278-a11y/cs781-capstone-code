# Report correction - 15 November 2026

The Phase 4 stacking explanation now matches the preserved executable builders:

1. Read two raw class logits per base model; align case IDs.
2. Form the class-1 minus class-0 logit margin for each model.
3. Standardize each margin column with StandardScaler fitted on training-calibration rows.
4. Apply L2 logistic regression, sigmoid, and the recorded decision threshold.

The recorded coefficients act on standardized margins. Cross-validation fits both scaler and classifier inside each training fold. The final pipeline is fitted on all 512 training-calibration rows and reused on official dev/test. No model training, calibration, predictions, or historical scores were changed by this documentation correction.

The combined PDF and LaTeX source are corrected. All three Phase 4 stack READMEs, technical references, reports, illustrated notebooks, diagrams, and downloadable packages receive the same correction. The public GitHub link remains unchanged.

## Why the combined report still says DRAFT

The student's own Q4 contribution statement is still unfinished. It must describe the student's actual work; this correction does not invent that statement. Phase 3 is accurately described as prepared but unrun. The retained draft label identifies an unfinished report; it does not mean the corrected methodology is awaiting a code change.

This revision updates GitHub materials only. It is not evidence of replacing or verifying a file uploaded to Hello IITK.
