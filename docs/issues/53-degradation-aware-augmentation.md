# Synthetic Degradation and Augmentation for Restoration Generalization

## Objective

Review literature on synthetic degradation pipelines, augmentation strategies, and restoration generalization, with particular attention to their relevance for SEM image restoration.

The research provides evidence-based context for Issues #40 and #41.

---

## 1. Synthetic Degradation

Synthetic degradation pipelines are commonly used to generate degraded training images from clean references.

### Gaussian Noise

The reviewed restoration literature commonly synthesizes additive Gaussian noise using either:

- Fixed noise levels.
- Randomly sampled noise levels.
- Ranges of noise severity.

For example, DnCNN-based experiments use fixed Gaussian noise levels, while other restoration pipelines sample noise severity from a range to represent multiple degradation conditions.

### Evidence Classification

**Direct Literature Evidence**

---

### Speckle / Multiplicative Noise

KLA confirms that the project benchmark includes multiplicative speckle noise.

The available project analysis indicates that the benchmark's noisy images can contain values outside the normal image range, with observed values extending approximately from `-0.27` to `1.93`.

This provides project-specific evidence that the benchmark's degradation process can produce interactions between additive and multiplicative components.

However, the available literature does not establish a specific synthetic speckle-generation strategy that should be adopted for this benchmark.

### Evidence Classification

**KLA/Project-Confirmed**

**Project-Specific Inference**

---

### Downsampling

Downsampling is a common component of synthetic degradation pipelines for super-resolution and restoration.

Bicubic interpolation is commonly used to generate low-resolution training pairs.

KLA also confirms downsampling as one of the degradation mechanisms used by the benchmark.

### Evidence Classification

**Direct Literature Evidence**

**KLA/Project-Confirmed**

---

### Blur + Downsampling / PSF Modeling

The reviewed literature emphasizes that a synthetic forward degradation model should resemble the physical imaging system when possible.

Applying blur before downsampling can provide a more physically motivated degradation process by approximating the effect of a Point Spread Function (PSF).

This evidence comes primarily from super-resolution and imaging research and should not automatically be interpreted as proof that a particular blur kernel represents the SEM acquisition process.

### Evidence Classification

**Direct Literature Evidence**

**Qualified for SEM Application**

---

## 2. Degradation Combinations and Ordering

Multiple degradation processes can be combined into synthetic degradation pipelines.

The reviewed literature demonstrates that changing the forward degradation process can affect restoration performance.

For example, changing degradation parameters such as the Gaussian filtering configuration can reduce restoration quality when the training and testing degradation processes differ.

This supports the broader observation that restoration models can be sensitive to the degradation distribution used during training.

KLA confirms that Gaussian noise, speckle noise, and downsampling are benchmark degradation mechanisms and that their ordering may vary.

The available sources do not establish a single optimal degradation order for the project benchmark.

### Evidence Classification

**Direct Literature Evidence**

**KLA/Project-Confirmed**

---

## 3. Degradation Severity

### Fixed vs. Randomized Severity

The reviewed literature uses both fixed degradation levels and randomized severity ranges.

Randomized severity allows a restoration model to encounter multiple degradation strengths during training instead of being exposed to only one fixed level.

However, the available sources do not contain a direct ablation demonstrating that randomized severity is universally superior to fixed-severity training for generalization on this SEM benchmark.

Therefore, randomized severity should be treated as a literature-supported strategy rather than a proven requirement for this project.

### Evidence Classification

**Qualified Literature Evidence**

---

## 4. Augmentation Strategies

The reviewed literature demonstrates that augmentation can affect restoration performance and generalization, but the evidence is strongly dependent on the dataset, task, and type of augmentation.

### Horizontal and Vertical Flips

In NAFSSR stereo super-resolution experiments on Flickr1024, horizontal and vertical flips produced measurable PSNR improvements.

These results demonstrate that geometric augmentation can be beneficial for the tested natural-image restoration task.

They do not directly prove the same improvement on the project's SEM benchmark.

### Evidence Classification

**Direct Natural-Image Evidence**

---

### Orthogonal Rotations

The NAFSSR experiments also evaluated orthogonal rotations as part of geometric augmentation.

The results support the use of these transformations in the tested stereo super-resolution setting.

For the project, orthogonal rotations are considered compatible with the pixel-grid structure of the SEM images because they do not require interpolation when implemented as exact 90-degree rotations.

This SEM applicability is a project-specific consideration rather than direct experimental evidence of improved benchmark performance.

### Evidence Classification

**Direct Natural-Image Evidence**

**Project-Specific Inference for SEM**

---

### Channel Shuffling

Channel shuffling produced measurable improvements in the cited RGB stereo-super-resolution experiments.

However, the project uses single-channel grayscale SEM imagery, making RGB channel shuffling physically irrelevant to the benchmark.

Therefore, the natural-image result should not be generalized to this project.

### Evidence Classification

**Direct Natural-Image Evidence**

**Project-Specific Limitation**

---

### Stochastic Depth

Stochastic depth improved out-of-distribution performance in the cited NAFSSR experiments.

The reported improvement was approximately `+0.16 dB` for the tested large model.

However, the experiment was conducted in a natural-image stereo-super-resolution setting and does not establish the same benefit for the project's NAFNet configurations.

### Evidence Classification

**Direct Natural-Image Evidence**

**Qualified for Project Application**

---

## 5. Synthetic-to-Real Gap and Distribution Shift

### Synthetic-to-Real Differences

Synthetic degradation assumes that the degradation model used during training adequately represents the degradation encountered during evaluation.

If the synthetic forward model differs substantially from the real degradation process, restoration performance can decrease.

The reviewed literature therefore highlights the importance of making synthetic degradation models representative of the intended imaging process.

### Evidence Classification

**Direct Literature Evidence**

---

### Acquisition Distribution Shift in SEM

SEM restoration research demonstrates that models can experience notable performance drops when acquisition conditions change.

Reported changes include factors such as:

- Beam alignment
- Focus
- Accelerating voltage

This demonstrates that restoration models can be sensitive to distribution shifts even when the general imaging modality remains the same.

### Evidence Classification

**Direct SEM/Microscopy Evidence**

---

### Hidden-Test Distribution Shift

KLA confirms that the hidden evaluation data may contain image content and structures that differ substantially from the training images.

KLA also indicates that degradation severity may differ between training and hidden-test data.

Therefore, the benchmark evaluates generalization beyond simply reproducing the training examples.

However, this does not prove that a particular augmentation strategy will solve the distribution-shift problem.

### Evidence Classification

**KLA/Project-Confirmed**

---

## 6. Relevance to Issues #40 and #41

The literature provides several findings that can inform the project's augmentation and generalization investigations.

### Benchmark Degradation Mechanisms

KLA confirms that the benchmark uses:

- Gaussian noise
- Speckle noise
- Downsampling

These mechanisms should remain the primary project context when interpreting the literature.

Published degradations should not automatically be added simply because they appear in other restoration pipelines.

### Severity Variation

Literature demonstrates the use of randomized degradation severity.

This provides a potential research direction for studying robustness to different degradation strengths, but the benefit for this benchmark remains experimentally unverified.

### Geometric Augmentation

Natural-image restoration research demonstrates benefits from flips and rotations.

Project documentation considers exact horizontal/vertical flips and orthogonal rotations compatible with the pixel-grid structure of the SEM imagery.

Whether these transformations improve performance on the hidden SEM test distribution remains an empirical question.

### Physical Degradation Modeling

Literature supporting physically motivated forward models provides context for considering whether synthetic degradation resembles the intended acquisition process.

This does not establish that every published degradation should be included in the project.

### Evidence Classification

**Direct Literature Evidence**

**KLA/Project-Confirmed**

**Project-Specific Inference**

---

## 7. Research Gaps

### Fixed vs. Variable Severity

The performance difference between training using the benchmark's exact degradation parameters and training using a distribution of degradation severities has not been established experimentally for this dataset.

### Multiplicative Speckle Generalization

The available literature provides limited evidence about how synthetic multiplicative speckle noise affects the synthetic-to-real gap specifically for SEM imagery.

### Degradation Ordering

The exact sequential ordering of degradation mechanisms used for the official hidden evaluation data remains undisclosed.

No optimal degradation ordering has been established.

### Content-Specific Augmentation

There is no evidence in the available sources demonstrating that specific content-based augmentations, such as simulating semiconductor defects or structures, improve performance on the hidden test set.

### Augmentation Effectiveness on SEM

Although geometric augmentations improve results in some natural-image restoration experiments, their quantitative effect on this specific SEM benchmark remains unknown.

### Evidence Classification

**Unknown / Insufficient Evidence**

---

## 8. Evidence Classification

The findings in this issue are separated into four categories.

### Direct Literature Evidence

The source directly demonstrates or reports the finding.

Examples:

- Gaussian noise synthesis.
- Geometric augmentation improvements on Flickr1024.
- Stochastic-depth improvements in NAFSSR.
- Distribution-shift effects in SEM restoration.
- Physically motivated degradation modeling.

### Qualified Evidence

The literature supports the general concept, but the evidence is limited by dataset, task, or experimental conditions.

Examples:

- Randomized degradation severity.
- Applying natural-image augmentation findings to SEM.
- Synthetic-to-real conclusions from non-SEM restoration tasks.

### KLA / Project-Confirmed

The project documentation or KLA guidance directly establishes the project context.

Examples:

- Gaussian noise, speckle noise, and downsampling are benchmark degradations.
- Hidden-test image content may differ substantially from training content.
- KLA recommends augmentation as one possible generalization strategy.

### Project-Specific Inference

The project makes an interpretation based on the available evidence.

Examples:

- Exact flips and 90-degree rotations being suitable for the SEM pixel grid.
- Considering benchmark-specific degradation severity distributions.

These should not be presented as experimentally proven improvements.

---

## 9. Key Takeaways

1. **Synthetic degradation must represent the intended problem:** Restoration performance can decrease when the training degradation process differs from the evaluation degradation process.

2. **Degradation severity matters:** Literature commonly uses multiple degradation strengths, but the benefit of randomized severity over fixed severity remains unverified for this benchmark.

3. **Degradation ordering can matter:** Changing the forward degradation process can affect restoration performance, but no optimal ordering is established for the project's hidden test set.

4. **Augmentation results are task-dependent:** Flips, rotations, and stochastic depth have demonstrated benefits in natural-image restoration experiments, but those results do not directly establish equivalent improvements for SEM.

5. **SEM models are sensitive to distribution shifts:** Changes in acquisition conditions such as focus, beam alignment, or accelerating voltage can reduce restoration performance.

6. **Content distribution is an important project concern:** KLA expects hidden-test image structures and content to differ from the training distribution.

7. **Published degradations should not be adopted automatically:** A degradation or augmentation used successfully in another restoration task does not necessarily represent the physical SEM benchmark.

8. **Project-specific validation is required:** The effectiveness of severity sampling and augmentation strategies on this SEM benchmark remains an empirical question.

---

## Conclusion

The literature establishes that synthetic degradation design and augmentation can influence restoration quality and generalization, but their effectiveness depends strongly on the degradation model, dataset, imaging process, and distribution being evaluated.

For this project, the confirmed benchmark degradations are Gaussian noise, speckle noise, and downsampling. The literature provides context for studying degradation severity, physically motivated degradation models, and geometric augmentation, but does not establish that any particular additional degradation or augmentation will improve the hidden SEM benchmark.

The main unresolved question for Issues #40 and #41 is therefore not which published augmentation should automatically be adopted, but which strategies are supported strongly enough to justify empirical evaluation on the project's specific SEM distribution.