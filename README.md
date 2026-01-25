# STRIDER: Navigation via Instruction-Aligned Structural Decision Space Optimization

> **NeurIPS 2025**  
> **Authors:** Diqi He, Xuehao Gao, Hao Li, Dingwen Zhang, Junwei Han

##  Abstract

The Zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE) task requires agents to navigate previously unseen 3D environments using natural language instructions, without any scene-specific training. A critical challenge in this setting lies in ensuring agents’ actions align with both spatial structure and task intent over long-horizon execution. Existing methods often fail to achieve robust navigation due to a lack of structured decision-making and insufficient integration of feedback from previous actions. To address these challenges, we propose STRIDER (Instruction-Aligned Structural Decision Space Optimization), a novel framework that systematically optimizes the agent’s decision space by integrating spatial layout priors and dynamic task feedback. Our approach introduces two key innovations: 1) a Structured Waypoint Generator that constrains the action space through spatial structure, and 2) a Task-Alignment Regulator that adjusts behavior based on task progress, ensuring semantic alignment throughout navigation. Extensive experiments on the R2R-CE and RxR-CE benchmarks demonstrate that STRIDER significantly outperforms strong SOTA across key metrics; in particular, it improves Success Rate (SR) from 29\% to 35\%, a relative gain of 20.7\%. Such results highlight the importance of spatially constrained decision-making and feedback-guided execution in improving navigation fidelity for zero-shot VLN-CE.

## TODOs

* [X] Release the code
* [ ] Update repository & homepage

## Installation & Data

Please refer to [Open-Nav](https://github.com/YanyuanQiao/Open-Nav/tree/main). 

## Inference

To run inference, use the script:

```bash
bash run_OpenNav.bash
```

