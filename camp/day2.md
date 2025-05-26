# day2
this note contains 
- my observations, ideas and stream of conciseness 
- reflection on them
- plans on next steps

materials of the day

# adversarial attack

https://colab.research.google.com/drive/1rHI5WlIGllsDzRULDcyFb8GCyP_7i8vU?usp=sharing

- model.eval() -- makes **deterministic** outputs
- no_grad
- require grad = false

# Risks discussion
- s risks scenarios can easily turn into x risk scenarios by trying to solve the problem with AI 

- how to produce usefully scenario activities: 
  - imagine a concrete scenario
  - divide it into steps if not done before
  - start from a beginning and estimate how many equally dangerous alternatives you can name
  - repeat to each step, making a tree
  - estimate a resulting probability\likelihood or whatever you want

# Transformers 
https://colab.research.google.com/drive/10G7VQeQqn2OcaySrTyYzsNc62VEFbdB6

- Recommended:
  - Transformers (how LLMs work) explained visually | DL5
https://www.youtube.com/watch?v=wjZofJX0v4M&
  - Attention in transformers, step-by-step | DL6
https://www.youtube.com/watch?v=eMlx5fFNoYc
- Optional:
  - How might LLMs store facts | DL7 
  https://www.youtube.com/watch?v=9-Jl0dxWQs8


- Exercise:

  - transformer_hard, https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/workshops/transformer/transformer_hard.ipynb
  - transformer_normal https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/workshops/transformer/transformer_normal.ipynb

# concept-map
- logits 

*logits are unnormalized, raw "scores" of the model for each class. they are converted to probabilities via softmax*

- language to vectors = lang 2 sub units + sub units 2 vecs
- one-hot encodings

*one-hot encoding converts a category into a vector, where a single "1" indicates which category is selected, and all other positions are filled with zeros*



- positional embeddings

In the classic Transformer, in addition to its “semantic” embedding, a positional embedding is added to the token so that the model “knows” where exactly each token is in the sequence.

- attention
- attention is all we need but it is a mass =)
------
# takeaways 
- meme contests are cool. do meme contests

# ideas 
- it would be cool to create a course there one builds NN step by step: classifier - hand -written number recognition - adverserial attacks - etc. It could kind of help keep the whole picture in one's mind
