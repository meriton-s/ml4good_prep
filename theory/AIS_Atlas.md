
# AI Safety Atlas — Chapter 01: Capabilities 
https://ai-safety-atlas.com/chapters/01/

## Quotes

> what AI systems can currently do, how they achieve these capabilities,  how we might anticipate their future development.

> the discussion of dangerous capabilities and potential risks (Chapter 2) follows directly from understanding capabilities.


> how AI systems have evolved from narrow, specialized tools to increasingly general-purpose tools.
## Document Structure

### 1.1 Introduction 
explains why it’s crucial to understand current AI **capabilities** for safety discussions. quotes Yann LeCun on the inevitability of superhuman AI. the chapter’s goal is to lay the groundwork: what modern systems can already do, how they achieve it, and where they might go next. it also introduces the idea that we should talk about **abilities** rather than abstract “intelligence,” so we can discuss risks concretely.

#### my thoughts  
- love the shift from “intelligence” to **capabilities**. but I think it is good for "what we do with what we have" question, not the engineering mindset like "how does it work"

### 1.2 State-of-the-Art AI
a survey of breakthrough achievements across different domains. shows the evolution from narrow, specialized models to more general systems. examples: language models that can reason; vision models with deep image understanding; advances in robotics (real-world training, growing autonomy); AI successes in games. this section demonstrates how rapidly AI capabilities have grown on all fronts (as of March 2024).
#### my thoughts
ok it is told that by early 2024 we have: 
- LLMs that reason beyond autocomplete  
- vision models with deep scene understanding  
- robots learning in real environments  
- game-playing agents beating humans  

curious how robotics progress rates compare to LLMs in curves 

the point is to show how fast narrow AIs have broadened their reach.
### 1.3 Foundation Models

shifts to the “foundation model” paradigm. describes the rise of huge, general-purpose models (e.g. large language models) replacing many small narrow ones. these “foundations” are trained on massive data in a self-supervised way and then fine-tuned for specific tasks. introduces **emergent capabilities**—unexpected skills that appear only at large scale. emphasizes that foundation models have dramatically expanded AI’s applicability but also introduced new safety challenges (harder to control, difficult to interpret, surprising failure modes).

#### my thoughts
was it the only possible way or a situational choice? why llm? 

### 1.4 Understanding Intelligence 
attempts to pin down what “general intelligence” means in practice and how to measure AI’s skills. the authors critique vague AGI definitions (“a system that outperforms humans at most tasks”) as unhelpful. they propose replacing the narrow vs. general dichotomy with **measurable abilities**: tallying how many domains and at what level a machine matches human performance. they introduce finer-grained metrics, for example a 2D view with axes for **performance** and **breadth** (how wide a range of tasks is covered). the goal is a clear scale to track progress toward “human-level” AI and trigger safety measures in time.

### 1.5 Scaling
discusses the role of sheer scale. invokes Richard Sutton’s “bitter lesson”: simple algorithms plus massive compute and data often beat clever, small systems. covers **scaling laws**—empirical relationships showing predictable quality gains as model size, data volume, and compute increase. graphs illustrate smooth improvements until new bottlenecks appear. also debates whether brute-force scaling (the strong scaling hypothesis) alone can yield transformative AI or if new ideas will eventually be needed. this section helps us understand the trajectory of model power and where sudden “jumps” or plateaus might occur.

### 1.6 Forecasting 
explores methods for predicting future capabilities. building on current trends and scaling laws, the authors examine techniques for estimating when AGI/ASI might emerge. they discuss “biological anchors”—attempts to gauge how much compute is required to match a human brain or evolutionary processes—and trend extrapolations (compute growth, model size, benchmark performance). the aim is to provide tools to forecast timelines for powerful AI and prepare safety measures in advance. ```

#### my thoughts
human brain is very much of an architecture not the general pattern it seems to be a bad anchor, no?