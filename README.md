# Neural-Machine-Translation
Translation of human-readable dates ("25th of June, 2009") into machine-readable dates ("2009-06-25").

This is implemented using an attention model, one of the most sophisticated sequence-to-sequence models.

## Translating human readable dates into machine readable dates

- This model will build here could be used to translate from one language to another, such as translating from English to Telugu.
- The network will input a date written in a variety of possible formats (e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987").
- The network will translate them into standardized, machine readable dates (e.g. "1958-08-29", "1968-03-30", "1987-06-24").
- We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD.
## Dataset
We will train the model on a dataset of 10,000 human readable dates and their equivalent, standardized, machine readable dates.

Preprocess the data and map the raw text data into the index values:
1. Set Tx=30.
  - Tx is the maximum length of the human readable date.
  - If we get a longer input, we would have to truncate it.
2. Set Ty=10
  - "YYYY-MM-DD" is 10 characters long.

- X: a processed version of the human readable dates in the training set.
  - Each character in X is replaced by an index (integer) mapped to the character using human_vocab.
  - Each date is padded to ensure a length of  Tx  using a special character (< pad >).
  - X.shape = (m, Tx) where m is the number of training examples in a batch.
- Y: a processed version of the machine readable dates in the training set.
  - Each character is replaced by the index (integer) it is mapped to in machine_vocab.
  - Y.shape = (m, Ty).
- Xoh: one-hot version of X.
  - Each index in X is converted to the one-hot representation (if the index is 2, the one-hot version has the index position 2 set to 1, and the remaining positions are 0.
  - Xoh.shape = (m, Tx, len(human_vocab)).
- Yoh: one-hot version of Y.
  - Each index in Y is converted to the one-hot representation.
  - Yoh.shape = (m, Tx, len(machine_vocab)).
  - len(machine_vocab) = 11 since there are 10 numeric digits (0 to 9) and the - symbol.

## Neural machine translation with attention
- If you had to translate a book's paragraph from French to English, you would not read the whole paragraph, then close the book and translate.
- Even during the translation process, you would read/re-read and focus on the parts of the French paragraph corresponding to the parts of the English you are writing down.
- The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step.

### Attention mechanism
<table>
  <tr>
    <td><img src="/images/attn_model.png" width=470 height=480></td>
    <td><img src="/images/attn_mechanism.png" width=470 height=480></td>
  </tr>
</table>

- The diagram on the left shows the attention model.
- The diagram on the right shows what one "attention" step does to calculate the attention variables α<sup>⟨t,t′⟩</sup>.
- The attention variables α<sup>⟨t,t′⟩</sup> are used to compute the context variable  context<sup>⟨t⟩</sup>  for each timestep in the output (t=1,…,T<sub>y</sub>).
#### Pre-attention and Post-attention LSTMs on both sides of the attention mechanism
- There are two separate LSTMs in this model (see diagram on the left): pre-attention and post-attention LSTMs.
- Pre-attention Bi-LSTM is the one at the bottom of the picture is a Bi-directional LSTM and comes before the attention mechanism.
  - The attention mechanism is shown in the middle of the left-hand diagram.
  - The pre-attention Bi-LSTM goes through  T<sub>x</sub>  time steps.
- Post-attention LSTM: at the top of the diagram comes after the attention mechanism.
  - The post-attention LSTM goes through  T<sub>y</sub>  time steps.
- The post-attention LSTM passes the hidden state s<sup>⟨t⟩</sup> and cell state c<sup>⟨t⟩</sup> from one time step to the next.
#### Each time step does not use predictions from the previous time step
- In this model, the post-attention LSTM at time t does not take the previous time step's prediction y<sup>⟨t−1⟩</sup> as input.
- The post-attention LSTM at time 't' only takes the hidden state s<sup>⟨t⟩</sup> and cell state c<sup>⟨t⟩</sup> as input.
- We have designed the model this way because unlike language generation (where adjacent characters are highly correlated) there isn't as strong a dependency between the previous character and the next character in a YYYY-MM-DD date.
#### Computing "energies"  e<sup>⟨t,t′⟩</sup>  as a function of  s<sup>⟨t−1⟩</sup>  and  a<sup>⟨t′⟩</sup>
- "e" is called the "energies" variable.
- s<sup>⟨t−1⟩</sup> is the hidden state of the post-attention LSTM.
- a<sup>⟨t′⟩</sup>  is the hidden state of the pre-attention LSTM.
- s<sup>⟨t−1⟩</sup>  and a<sup>⟨t⟩</sup> are fed into a simple neural network, which learns the function to output e<sup>⟨t,t′⟩</sup>.
- e<sup>⟨t,t′⟩</sup>  is then used when computing the attention α<sup>⟨t,t′⟩</sup> that y<sup>⟨t⟩</sup> should pay to a<sup>⟨t′⟩</sup>.
  
  ## Visualizing Attention
Since the problem has a fixed output length of 10, it is also possible to carry out this task using 10 different softmax units to generate the 10 characters of the output. But one advantage of the attention model is that each part of the output (such as the month) knows it needs to depend only on a small part of the input (the characters in the input giving the month). We can visualize what each part of the output is looking at which part of the input.

Consider the task of translating "Saturday 9 May 2018" to "2018-05-09". If we visualize the computed  α<sup>⟨t,t′⟩</sup>  we get this:
<p align = 'center'>
  <img src = '/images/date_attention.png'>
</p>
Notice how the output ignores the "Saturday" portion of the input. None of the output timesteps are paying much attention to that portion of the input. We also see that 9 has been translated as 09 and May has been correctly translated into 05, with the output paying attention to the parts of the input it needs to to make the translation. The year mostly requires it to pay attention to the input's "18" in order to generate "2018."
