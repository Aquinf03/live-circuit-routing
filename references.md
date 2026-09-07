# Must-cite references 

Short list for the live routing-graph paper. Links preferred over perfect BibTeX for now.

## Attention circuits / foundations

1. **Elhage et al. (2021).** *A Mathematical Framework for Transformer Circuits.*
  [https://transformer-circuits.pub/2021/framework/index.html](https://transformer-circuits.pub/2021/framework/index.html)  
   → QK / OV; attention as communication channels.
2. **Olsson et al. (2022).** *In-context Learning and Induction Heads.*
  [https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)  
   → Ground-truth task #1 (induction).
3. **Wang et al. (2023).** *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.* ICLR.
  [https://arxiv.org/abs/2211.00593](https://arxiv.org/abs/2211.00593)  
   → Ground-truth task #2 (IOI) on GPT-2 Small.



## ACDC-style / patching discovery

1. **Conmy et al. (2023).** *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS.
  [https://arxiv.org/abs/2304.14997](https://arxiv.org/abs/2304.14997)  
   → ACDC; automated edge pruning via interventions.
2. **Heimersheim & Nanda (2024).** *How to use and interpret activation patching.*
  [https://arxiv.org/abs/2404.15255](https://arxiv.org/abs/2404.15255)  
   → Standard causal verification language for our ablations.



## Routing / information-flow graphs

1. **Ferrando & Voita (2024).** *Information Flow Routes: Automatically Interpreting Language Models at Scale.* EMNLP.
  [https://arxiv.org/abs/2403.00824](https://arxiv.org/abs/2403.00824)  
   → Closest neighbor: subgraph from attribution in ~one forward; we stay routing-first / attention-graph first.



## Attribution graphs / feature circuits / QK

1. **Ameisen, Lindsey, Pearce, et al. (2025).** *Circuit Tracing: Revealing Computational Graphs in Language Models.* Anthropic.
  [https://transformer-circuits.pub/2025/attribution-graphs/methods.html](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)  
   → Attribution graphs via replacement models (CLTs); heavy but gold-standard graphs.
2. **Anthropic (2025).** *Tracing Attention Computation Through Feature Interactions.*
  [https://transformer-circuits.pub/2025/attention-qk/index.html](https://transformer-circuits.pub/2025/attention-qk/index.html)  
   → QK attributions; explains *why* heads attend (feature×feature), after freezing patterns.
3. **Marks et al. (2025).** *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models.* ICLR.
  [https://arxiv.org/abs/2403.19647](https://arxiv.org/abs/2403.19647)  
   → Feature-level causal circuits (SAEs); what we are *not* claiming to replace.



## Attention gap / complete tracing (optional but useful)

1. **OpenMOSS / CRM line (2025).** *Bridging the Attention Gap: Complete Replacement Models for Complete Circuit Tracing.*
  [https://interp.open-moss.com/posts/complete-replacement](https://interp.open-moss.com/posts/complete-replacement)  
    → Shows attribution graphs still leave attention hard; motivates routing-as-first-class object.



## Extra if space

1. **Syed, Rager & Conmy (2023).** *Attribution Patching Outperforms Automated Circuit Discovery.*
  [https://arxiv.org/abs/2310.10348](https://arxiv.org/abs/2310.10348)  
    → Cheap attribution vs iterative discovery; useful baseline framing.
2. **Hanna, Liu & Variengien (2023).** *How does GPT-2 compute greater-than?*
  [https://arxiv.org/abs/2305.00586](https://arxiv.org/abs/2305.00586)  
    → Another known GPT-2 Small circuit if induction+IOI need a third check.

