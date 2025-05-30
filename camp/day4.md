# day4
## Governance
https://docs.google.com/presentation/d/1tDJ5h4epvR8pQSrDJjJbub3ZebaxK3zL3nF3-k3_aU4/edit?usp=sharing


National rules can’t tame AI because no country controls all research, threats spill across borders, and strict standards at home just push work abroad, starting a race to the bottom.
Even so, strong domestic moves—like the EU AI Act or U.S. chip export controls—reshape global behavior through the “Brussels Effect” and supply-chain leverage.
International coordination is picking up: the 2023 UK AI Safety Summit, the G7 Hiroshima process, pending UN and OECD work, a Council of Europe treaty draft, and China’s own governance plan.
We are only at the agenda-setting and early formulation stages of a five-step policy cycle that ends with global implementation and review.
Three playbooks dominate discussion: create new institutions for consensus, enforcement, and emergency response; apply non-proliferation controls on chips, data, and models; and forge regulatory pacts with verification tools like on-chip weight snapshots to ease the security-transparency trade-off.
A tougher backup is containment—e.g., the MAGIC plan to channel every frontier-scale training run into one multinational lab—but politics, trust, and shrinking compute thresholds make it unlikely for now.
Real progress will mix these tools; states join when risks feel imminent, compliance costs stay modest, secrets stay safe, big players lead, and smaller ones get help, balancing broad membership against strong rules.

### The Brussels Effect 
The Brussels Effect is a phenomenon whereby strict European Union rules are extended beyond the EU: global companies, in order to have access to the huge single market, implement these rules in all countries, and therefore the standards adopted in Brussels effectively become global.
### Key international steps to coordinate AI policy

**UK AI Safety Summit (Bletchley Park, 1-2 November 2023)**
28 countries, including the US and China, signed the "Bletchley Declaration". The document recognized the risks of "frontier" models and launched a network of joint research and regular checks of systems before their release; subsequent summits agreed to be held annually.

**G7 Hiroshima AI Process (May-October 2023)**
The G7 countries adopted international **Guiding Principles** and a voluntary **Code of Conduct** for developers of the most powerful models. The goal is common standards of transparency, risk management and information exchange without strict regulation.

**UN: High-Level Advisory Body on AI**
Created at the initiative of the Secretary-General in 2023. Published an interim report "Governing AI for Humanity" (December 2023) and a final report (September 2024), proposing a model for global computing monitoring and an incident registry; findings are discussed at the "Summit of the Future".

**OECD / GPAI**
The OECD has been leading the "Trustworthy AI" Principles since 2019; in July 2024, the GPAI initiative was formally merged with the OECD's work, creating a single platform for collecting incident data, trust metrics, and a new **AI Compute Task Force** to track capacity.

**Council of Europe: Framework Convention on AI**
The first legally binding convention on AI opened for signature on 5 September 2024. It requires states to assess risks, transparency, remedies, and covers both the public and private sectors; non-Council of Europe members are welcome to join (US, Canada, Japan, etc. have signed).

**China: Global AI Governance Initiative and Domestic Rules**
In October 2023, President Xi Jinping presented the **Global AI Governance Initiative**, calling for equitable treatment, model traceability, and consideration of the interests of the Global South. Domestically, the **Interim Measures for Generative AI Services** regulation (since August 2023) is already in force, establishing provider liability and mandatory content labeling.

## literature review 
https://docs.google.com/document/d/1gBid822YCB45KgoHFAgefMV5BKA1FvUt0IoZscNHbcA/edit?tab=t.0#heading=h.2qh9sibrrpic


# Developmental Interpretability (DevInterp):
_DevInterp adds a **time axis** to interpretability: it’s not just “What does this network do?” but also “**When** and **how** did those pieces appear?”_


> **Goal:** Watch how a neural network **grows** its inner parts while it is still learning, not only after training is done.

### What is developmental interpretability

> Instead of “deciphering” a ready-made model, DevInterp looks at the phases and phase transitions in the gradient descent process and tries to understand what chunks of computation “grow” at each stage. The methodology relies mainly on Singular Learning Theory (SLT) — a statistical geometric theory that explains why jumps between different “submodels” occur when training complex models.

* Looks at **checkpoints** saved during training.
* Tries to spot **“growth spurts”**—moments when a new skill or circuit suddenly appears.
* Treats training like a **time-lapse video** instead of a single photo.

## Why It Matters

* **Shorter(?) path to understanding** big models.  
* hypothetical possibility of **Early-warning system** for unsafe behaviors.  
* If succeeded could let us **edit training in real time**—adding data, regularizing, or freezing layers to stay on the safe road.

## Theory of success

- Reliably detect transitions — automatic metrics (RLCT, susceptibilities, etc.) signal exactly where new capabilities are born in the model.

- Link the transition to the structure — mechanistic interpretability tools launched at checkpoints before and after the jump show which particular scheme (attention head, layer, feature-SAE, etc.) has emerged.

- Form a "development map" of the model — an ordered log of such transitions, giving the "assembly history" of the network.

- Predict and manage — knowing where the network "turns the wrong way", the researcher can change the data, regularization, or freeze the layer to correct the trajectory.

- Pack everything into a working stack — the devinterp library + experiment recipes should become as standard as SAE audit or autoneural attribution search are today.

## Background claims
Dev Interp tries to build a chronology of the network's development. If phase transitions:

- exist and are repeatable (across different models);

- are localizable (a specific layer/head can be found);

- are predictable (metrics like RLCT flare up before a new ability appears),

- then the researcher gets "shooting points" for a training time-lapse. This simplifies the analysis of the internal logic of the model and gives a chance to intervene before unwanted behavior is born.

---
https://docs.google.com/presentation/d/1gPwNiJ2cy5ZnonRmn1cZH9GcGXtgVsyw_2LZLHrv30Q/edit?slide=id.g35ea29cd0c1_1_82#slide=id.g35ea29cd0c1_1_82


Dev interp: Towards Developmental Interpretability — LessWrong 
https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability


Site : devinterp.com
SLT : lesswrong.com/tag/singular-learning-theory
Last review : manifund.org/projects/next-steps-in-developmental-interpretability
Neural networks generalize because of this one weird trick - LessWrong
https://www.lesswrong.com/posts/fovfuFdpuEwQzJu2w/neural-networks-generalize-because-of-this-one-weird-trick
Distilling Singular Learning Theory - LessWrong
https://www.lesswrong.com/s/czrXjvCLsqGepybHC
Project Ideas 
https://timaeus.co/projects

## Towards Developmental Interpretability
by Jesse Hoogland, Alexander Gietelink Oldenziel, Daniel Murfet, Stan van Wingerden

> developmental interpretability:
> - is organized around phases and phase transitions as defined mathematically in SLT, and 
> - aims at an incremental understanding of the development of internal structure in neural networks, one phase transition at a time.

**"phase transitions" in network training**


The authors claim that during training, neural networks undergo sharp, abrupt changes in their internal structure and behavior - an analogue of "phase transitions" in physics (ice → water). For the developmental interpretability program, such intra-training jumps are especially important: if you catch them, you can understand when and what new "module" the network was born.

### QUESTIONS
 > We don't believe that all knowledge and computation in a trained neural network emerges in phase transitions, but our working hypothesis is that enough emerges this way to make phase transitions a valid organizing principle for interpretability. Validating this hypothesis is one of our immediate priorities.
 
WHY?