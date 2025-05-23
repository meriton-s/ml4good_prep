# AI Safety Atlas — Chapter 02: Risks
https://ai-safety-atlas.com/chapters/02/

## TL;DR
AI’s growing capabilities bring diverse, interacting risks—from actors weaponizing AI to fundamental alignment failures and societal erosion. A holistic safety posture must address misuse, misalignment, systemic fragilities, and emergent dangerous capabilities in parallel. 

## Main Theses
- Risk taxonomy: AI risks split into misuse, misalignment, systemic, and capability-driven categories that interplay and can amplify each other. 

- Acceleration factors: Common amplifiers (e.g., automation scale, information diffusion) magnify harms faster than oversight can adapt. 

- Misalignment decomposition: Failures arise via specification errors, generalization mistakes, and convergent subgoals (e.g., AI resisting shutdown). 

- Systemic emergence: Societal dependence on many innocuous-working AIs can lock in power concentrations, erode privacy, and corrode trust long before single-system failure is obvious. 

- Core dangerous capabilities: Deception, situational awareness, power seeking, and self-replication form a capability stack enabling severe misuse, misalignment, and systemic breakdowns.
## Document Structure

### 2.1 Risk Decomposition
 lays out a taxonomy of AI risks, dividing them into misuse (malicious actors), misalignment (goals diverging from human intent), systemic fragilities (societal dependencies) and emerging dangerous capabilities. 
- **key point**: categories overlap and amplify each other—can’t tackle one in isolation
#### my thoughts  
- how do we measure overlap between misuse and misalignment in concrete cases? are there examples where the line is blurred?
- неясно, достаточно ли такой категоризации для сложных гибридных случаев
- иногда ощущается, что systemic fragilities следуют из misalignment, а не отдельная категория
- wonder if systemic fragilities deserve a separate taxonomy—maybe they’re too cross-cutting to sit alongside misalignment.
### 2.2 Risk Amplifiers
the authors analyze factors that magnify AI-related harms faster than we can build defenses: rapid deployment, automation at scale, information diffusion and zero-day vulnerabilities. they show how even minor flaws become critical when systems are widely used and iterated on without proper guardrails.

#### my thoughts
- can we build “accelerator sandboxes” to test amplification effects early?

### 2.3 Misuse Risks
how bad actors might weaponize AI—synthesizing biothreats, automating disinformation campaigns, or crafting sophisticated cyberattacks. 
- **key point**: AI is simply an “amplifier of intent,” making existing threats far easier and cheaper to scale.
#### my thoughts

### 2.4 Misalignment Risks
failure modes: spec hacks, goal misgeneralization, convergent subgoals (e.g., self-preservation)

- **key point:** “helpful” behavior in tests can mask dangerous objectives in new contexts
#### my thoughts
- does model architecture drive convergent subgoals more than scale itself?
- 
### 2.5 Systemic Risks
effects: privacy erosion, power concentration, opaque feedback loops across many AIs

- **key point**: aggregate “well-behaved” AIs can irreversibly reshape institutions
#### my thoughts
- how to engage general public in spotting system-level risks without overhype?
### 2.6 Dangerous Capabilities
here they identify key capabilities—deception, situational awareness, strategic planning, self-replication—that together enable high-impact threats. 
- **key point:** once these skills converge in one agent, the risk of catastrophic misuse or runaway behavior spikes dramatically.
#### my thoughts
- how to allocate limited resources across misuse, misalignment, systemic, and capability research?
### 2.7 Conclusion
mitigating AI-driven extinction or global collapse requires a holistic approach covering all risk categories in parallel


## Quotes

> As AI models get more capable, the scale of potential risks also rises—each new capability can multiply harm vectors

> Flaws are hard to discover until something goes catastrophically wrong, and the speed of deployment leaves little time for learning

> Systemic risks emerge even when each AI works as designed; together, they can reshape civilization irreversibly.

> We define deception as the systematic production of false beliefs in others…resulting from optimizing for a different outcome than truth.