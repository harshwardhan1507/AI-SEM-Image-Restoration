# Model Capacity, Compute and Inference Efficiency

## Objective

Review research concerning the relationship between restoration model capacity, restoration quality, computational cost, and inference efficiency.

The research provides literature context for the project's NAFNet capacity-scaling work in Issue #38.

---

## 1. Model Capacity and Restoration Quality

Model capacity can be increased through factors such as:

* Network depth
* Network width
* Number of parameters
* Number of computational operations

The reviewed literature demonstrates that increasing capacity can improve restoration quality, but the relationship is not necessarily linear.

### Depth Scaling

NAFNet ablation studies show that increasing the number of blocks from 9 to 36 produced significant PSNR improvements on the tested restoration tasks.

For example, on the GoPro deblurring dataset:

* 9 blocks: 31.79 dB
* 36 blocks: 32.85 dB

The same experiments also demonstrated that increasing depth beyond this range does not necessarily provide proportional quality improvements.

### Width Scaling

The NAFSSR family demonstrates that increasing model size and representational capacity can improve restoration performance across its tested variants.

The reported variants range from approximately:

* Tiny: 0.46M parameters
* Large: 23.83M parameters

The results demonstrate that capacity scaling can improve performance in the tested stereo-super-resolution setting.

### Important Limitation

These scaling results were obtained on specific datasets and tasks.

They do not establish that a particular model width or depth is optimal for this project's SEM benchmark.

### Evidence Classification

**Direct Literature Evidence**

---

## 2. Computational Cost

Model capacity affects computational requirements, but parameter count alone does not completely describe computational cost.

Common efficiency measurements include:

* Parameter count
* FLOPs
* MACs
* GPU memory
* Training time
* Inference latency
* Throughput

### Parameters

Parameter count describes the number of learned values in a model.

The project considers multiple NAFNet capacity configurations, including a previously discussed approximately 67.8M parameter configuration.

This configuration is a candidate experiment rather than a predetermined optimal model.

### MACs / FLOPs

MACs and FLOPs provide estimates of computational work required by a model.

However, computational-operation counts should not automatically be treated as equivalent to real-world inference time.

### Evidence Classification

**Direct Literature Evidence + Project Context**

---

## 3. Compute vs. Real-World Latency

The reviewed SEM benchmarking evidence demonstrates that MACs and parameter counts do not always predict real-world inference latency directly.

For example, on 1024×768 SEM inputs and the reported RTX 6000 Ada setup:

* NAFNet: 26.70M parameters, 158.92G MACs, approximately 0.32s latency
* HINet: 88.67M parameters, 170.73G MACs, approximately 0.30s latency

HINet therefore achieved slightly lower measured latency despite having more parameters and higher reported MACs.

This demonstrates that factors beyond parameter count and MACs affect actual runtime.

These factors can include:

* Hardware
* Operators used
* Memory access
* Implementation efficiency
* Input size
* Execution configuration

### Evidence Classification

**Direct SEM Evidence**

---

## 4. GPU Memory and Training Cost

Model capacity also affects training resource requirements.

The reviewed project documentation and technical sources indicate that larger restoration models generally require greater computational and memory resources.

A reported NAFNet configuration using 256×256 patches and mixed precision was estimated to require approximately 4.5 GB of VRAM.

This value is configuration-dependent and should not be treated as a universal memory requirement for NAFNet.

### Evidence Classification

**Direct Technical Evidence**

---

## 5. Diminishing Returns

One of the most important findings for capacity scaling is that additional model capacity can eventually produce smaller quality improvements relative to its computational cost.

### NAFNet Depth Example

On the GoPro natural-image deblurring dataset, increasing NAFNet depth from 36 to 72 blocks produced:

* PSNR: 32.85 → 32.88 dB
* Improvement: approximately +0.03 dB
* Latency: 177.1 ms → 230.1 ms
* Latency increase: approximately 30%

The additional capacity therefore produced a very small quality improvement while increasing latency substantially.

The NAFNet study identified the 36-block configuration as providing a better performance/latency balance **for that experiment**.

This should not be interpreted as evidence that 36 blocks is optimal for SEM restoration.

### Smaller Models

The NAFSSR literature also demonstrates that smaller models can achieve competitive performance in their respective tasks.

NAFSSR-T achieved strong results with substantially fewer parameters than some previous approaches.

### Evidence Classification

**Direct Literature Evidence**

---

## 6. NAFNet Architectural Efficiency

The NAFNet architecture introduces several design choices intended to reduce unnecessary computation while maintaining restoration performance.

### SimpleGate

SimpleGate replaces conventional nonlinear activation operations with a gating mechanism based on element-wise multiplication of feature groups.

The NAFNet ablation studies demonstrate the effectiveness of this design within the tested architecture.

### Simplified Channel Attention

NAFNet uses Simplified Channel Attention to provide channel-wise feature scaling without the more complex operations used in traditional attention mechanisms.

The reported ablation results demonstrate that this simplified design can perform effectively for image restoration.

### Depthwise Convolutions

Depthwise convolutions reduce the number of parameters and computational operations compared with standard convolutions for the same channel configuration.

### Additive Skip Connections

NAFNet uses additive skip connections rather than concatenation in relevant parts of the architecture.

The reported architecture analysis indicates that this reduces memory requirements compared with feature concatenation.

### Important Limitation

These architectural properties demonstrate design-level efficiency mechanisms.

They do **not** establish that NAFNet will always be faster or more memory-efficient than every alternative restoration architecture or transformer.

Actual performance depends on implementation, hardware, input size, and execution conditions.

### Evidence Classification

**Direct Literature Evidence**

---

## 7. Quality vs. Compute Trade-off

The literature supports evaluating restoration models using both quality and efficiency rather than maximizing model capacity independently.

Increasing capacity can provide quality improvements, but those improvements may eventually become small relative to the additional:

* Parameters
* MACs/FLOPs
* GPU memory
* Training time
* Inference latency

The 36→72 block experiment provides a concrete example of this diminishing-return behavior.

However, the point at which diminishing returns begin is task- and dataset-dependent.

### Evidence Classification

**Strongly Supported Interpretation**

---

## 8. Inference Efficiency

Restoration efficiency can be evaluated using:

* Inference latency
* Throughput
* Frames per second
* Total processing time
* Computational complexity

The reviewed SEM literature reports substantial reductions in acquisition or processing time using AI-based restoration.

One reported SEM study demonstrated a large speed advantage compared with traditional multi-frame acquisition.

However, reported speedups depend on the exact acquisition method, hardware, image size, model, and processing pipeline.

Therefore, these values should be treated as results from the cited experiment rather than universal performance expectations.

### Evidence Classification

**Direct SEM Evidence**

---

## 9. KLA / Project Context

KLA evaluates **end-to-end inference runtime**, rather than only the neural-network forward-pass time.

The measured pipeline includes:

1. Image loading
2. Preprocessing
3. GPU transfer
4. Model inference
5. Post-processing
6. Writing restored images to disk

Therefore, model-only latency is only one component of the total runtime relevant to the project.

KLA also recommends seeking a strong quality/latency trade-off rather than simply maximizing model size.

### Evidence Classification

**KLA/Project-Confirmed Fact**

---

## 10. Relevance to Issue #38

Issue #38 evaluates NAFNet capacity scaling.

The project implementation allows capacity to be varied through parameters such as:

* Base channel width
* Number of blocks
* Stage configuration

A previously discussed approximately 67.8M parameter / Width=64 configuration is identified as a **candidate experiment**.

It should not be treated as:

* The optimal model size
* The required target
* A literature-established configuration for SEM
* Evidence that larger capacity will improve the benchmark

The literature instead supports experimentally comparing different capacity levels while recording both restoration quality and computational cost.

### Evidence Classification

**Project-Specific Context / Inference**

---

## 11. What the Literature Does Not Establish

### SEM Capacity Saturation

The reviewed NAFNet depth-scaling experiments demonstrate diminishing returns on natural-image datasets such as GoPro.

They do not establish the capacity at which diminishing returns begin for semiconductor SEM structures or the project's benchmark.

### Capacity and Multiplicative Speckle

The available literature does not isolate how increasing model capacity affects restoration of the benchmark's specific multiplicative speckle degradation.

This relationship requires project-specific experimentation.

### Batch and Tiled Throughput

There is limited evidence establishing how different batching strategies affect end-to-end throughput for high-resolution SEM images processed using tiled inference.

### KLA Quality vs. Runtime Relationship

KLA's exact relationship between restoration quality metrics and end-to-end runtime in the final evaluation is not publicly established in the available sources.

The project should therefore not invent a mathematical quality/latency weighting formula.

### Evidence Classification

**Unknown / Insufficient Evidence**

---

## 12. Capacity Evaluation Considerations

The literature suggests that capacity experiments should not evaluate model size using PSNR or SSIM alone.

A useful comparison should consider both restoration quality and resource requirements, including:

| Category            | Example Measurements                            |
| ------------------- | ----------------------------------------------- |
| Model capacity      | Parameters, width, depth                        |
| Computational cost  | MACs, FLOPs                                     |
| Memory              | GPU VRAM usage                                  |
| Training cost       | Training time, resources                        |
| Inference           | Latency                                         |
| Throughput          | Images/sec or tiles/sec                         |
| Restoration quality | PSNR, SSIM, LPIPS                               |
| Overall trade-off   | Quality improvement relative to additional cost |

These measurements allow capacity increases to be evaluated in terms of whether the additional computational cost produces meaningful restoration improvements.

---

## 13. Evidence Classification

The findings in this issue are separated into four categories.

### Direct Literature Evidence

The source directly reports or demonstrates the finding.

Examples:

* NAFNet depth-scaling results
* NAFSSR capacity-scaling results
* SEM latency comparisons
* NAFNet architectural ablations

### Strongly Supported Interpretation

Multiple findings support the broader conclusion.

Example:

* Increasing capacity should be evaluated against the additional computational cost rather than treated as automatically beneficial.

### Project-Specific Inference

The literature provides context for a project decision but does not establish the result experimentally.

Examples:

* Using staged capacity experiments in Issue #38
* Treating the ~67.8M configuration as a candidate rather than a target

### Unknown / Insufficient Evidence

The available sources do not establish a reliable conclusion.

Examples:

* SEM-specific capacity saturation
* Capacity versus multiplicative-speckle restoration performance
* Exact KLA quality/runtime weighting
* Tiled throughput behavior under the project's conditions

---

## 14. Key Takeaways

1. **More capacity can improve restoration quality:** Width and depth scaling produced measurable improvements in the reviewed restoration experiments.

2. **Scaling is not necessarily linear:** The NAFNet 36→72 block experiment demonstrates that additional capacity can eventually produce very small quality improvements relative to its latency cost.

3. **MACs do not equal latency:** SEM benchmarking demonstrates that models with different parameter counts and MACs can have similar or even lower measured latency depending on their architecture and execution environment.

4. **Architectural design matters:** NAFNet uses design choices such as SimpleGate, depthwise convolutions, simplified channel attention, and additive skip connections to reduce computational or memory overhead within its architecture.

5. **Efficiency must be measured in context:** Actual inference performance depends on hardware, implementation, input size, and the complete processing pipeline.

6. **KLA evaluates end-to-end runtime:** Image loading, preprocessing, GPU transfer, model inference, post-processing, and saving all contribute to the relevant runtime.

7. **Capacity saturation is benchmark-specific:** The 36-block diminishing-return result from GoPro cannot be treated as the optimal capacity for the project's SEM benchmark.

8. **The ~67.8M model is only a candidate:** The project should empirically determine whether additional NAFNet capacity produces meaningful quality improvements relative to its computational cost.

9. **Quality and efficiency should be evaluated together:** Capacity experiments should record restoration quality alongside parameters, computational cost, memory usage, and actual runtime.

---

## Conclusion

The literature establishes that restoration model capacity can improve image-restoration quality, but increasing capacity also increases computational and resource requirements and may eventually produce diminishing returns.

NAFNet provides documented architectural mechanisms for reducing computation and memory requirements, while SEM benchmarking demonstrates that theoretical complexity measures such as MACs do not necessarily predict real-world latency.

For this project, the relevant question is therefore not simply how large the NAFNet model can become, but whether additional capacity produces meaningful restoration-quality improvements relative to its computational and end-to-end inference cost.

The available literature does not establish the optimal capacity for the project's SEM benchmark. That question belongs to the empirical capacity-scaling experiments in Issue #38.
