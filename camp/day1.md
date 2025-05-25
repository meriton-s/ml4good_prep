# day1
this note contains 
- my observations, ideas and stream of conciseness 
- reflection on them
- plans on next steps

materials of the day^
- [IT25] Shared Folder
https://drive.google.com/drive/u/0/folders/1t42HrmWydDewJpwbrG3RgoitZZrg83ay
- [IT25] Shared Notes
https://docs.google.com/document/d/1Gf65Pa4KUzyC7KcTe6aaWvOCcsJJP8gW8szzH0mnk_0/edit?tab=t.90fv9ob613lo


# HyperParameters vs Parameters
- **What is the difference between hyperparameters and parameters**

HP is smth we don't train (optimize) via GD

- **How do one choose HP?**

In practice, it is the matter of best practices. Sometimes somewhere one can know some theory to predict wht HP would be optimal. Sometimes this theory is bullshit but it gives predictions in this exatc situation. (All models are wrong, but some are usefull)
Otherwise one can bruteforce or test and guess them
So:
- - theory
- - best practices
- - test and guess
### my thoughts 
- it is what it looks like for a discipline to be pre-paradigmatic: there are not a lot of theory to learn (so you can reach the frontier quicker) but you need still to do work and in order to do that you need to get that information from somewhere. Here best practicies comes from. but sometimes BP are so bad that guess and test work better.

## coding: Hyperparameters
colab notebook:
https://colab.research.google.com/drive/1WdxrciSXTf_J7uZ30cDizJ9gLcqEVFTp#scrollTo=JgJ5pzoYnJqj

### Exercise 1: Impact of the optimizer
Retrain the model by using different hyperparameters. You can change them in the definitions in previous cells, but it is recommended that you put the code for modifying the values of hyperparameters in the cell below.

Try to see the impact of the following factors:
- > Use different batch sizes from 16 to 1024.
  
- I predict the existence of minimum and not in the middle
- > Use different values of the learning rate (for example, between 0.001 and 10), and see how these impact the training process.
- I predict that lower learning rate will keep making things better but with decreasing deminishing returns

- > Change the duration of the training by increasing the number of epochs.
- I predict that increasing the number of epochs will make things better 
- Use other optimizers, such as Adam or RMSprop.
- 
## technical debt
- finish the notebook
- do a visualisation of the influence
- do groking. it is a right place to do it


# Capabilities discussion

## ideas 
- LLMs are better in coding because the space of code is so much smaller than of natural level
- if that is true then it also would be true for analytic\synthetic languages? And artificial languages? And other structures with smaller space?
- it is interesting to think about all of this from the point of configuration space

## my thoughts
- maybe i should attempt something like a workshop on interpretability not to insult the thing i'm not familiar
- i should take a look on the papers of world models in LLM (golden bridge) + "on the biology of large language models" by Anthropic
- competitive pressure might work not as i use to think "if we don't do bad thing those awful guys we are competing with will do even a worse thing". But as "we don't want to lose to those awful (or any) guys we are competing with" 

# Intro to AI Safety
## Risks
- Balance disruption
- Bad attractor risks
- Singularity risks
- Absense of AI risks

## Solutions
- Fix what exists
- Build correct from the beginning
- Stop doing (bigger) AI's

## Skills
the world will down-rate you on the lack of the following
- networking
- fundraising
- public\research communication
- technical skills (low level\grounded skills)
- strategic\systemic views (high level understanding)
- meta-skills (research taste, time management, self reflection and open-mindedness)

# How to create AGI

### to do:
- google about figure 01 + openAI: they are doing speech reasoning in a voice controlled robot
- theory of mind benchmarks and self\others overlap
- mirror test: youtube, scholar, miriY


- compare 
  - voyager (Wang 2023) that use gpt generating instructions and code that implement them communicating via ApI (knowing the state of the game)
  - IGOR that use llm+rl

- what exactly is going on with alfaGo\kataGo and adversarials? the issue is that winning in go is such a complex landscape that it doesn't generalize well. but why is it so? is it something to do with sampling? or is it a nature of the task? 

- watch youtube about alphaEvolve

# Agents

- What is an agent made of
  - scaffolding: code surrounding an llm core
    - ReAct (reason+Act) (llm prompts itself or calling subroutines)
    - Example: Magnetic One
  - AlphaEvolve(evolutionary algorithm under the hud)
  - CoT: inp-otp promting, cot, selfconsistemcy with CoT -\- tree of thought
- Memory
  - content
  - techniques (RAG)
- how to build an agent
- create tool
- create scaffolding
- choose an LLM
- write the prompts

- evaluate the agent 
- iterating the above steps

- RL the Models on the agent outcomes

### ideas
- llm+rl worked, llm+evoAlg worked. what else can be mixed with llm?
- try to create a funny evals like vending-bench but about ai safety management

### to do
- google self-consistency with CoT and tree of thought

### exisize
- hard https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/workshops/agents/agents_hard.ipynb
- normal https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/workshops/agents/agents_normal.ipynb
- my copy https://colab.research.google.com/drive/1wWPQWRfOKDsYXGbB7LTXP44zBJKde1ZY#scrollTo=vBMbIBdR1HHo
#### takeaways
- if you give llm access to tools and context they become more agentic