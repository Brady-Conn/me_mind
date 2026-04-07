# Swishful Thinking: Do State Space Models Dream of Recursive Sheep?
**Overcoming Gradient Starvation in Continuous Hierarchical SSMs via Multiplicative State Fusion**

### Abstract
Standard flat autoregressive models face a fundamental temporal paradox: predicting the next character requires fast-ticking, high-friction updates, whereas tracking long-range grammatical syntax requires slow-ticking, low-friction memory. While stacking State Space Models (SSMs) into discrete hierarchies offers a theoretical solution, naively connecting fast and slow layers creates a "Lazy Shortcut" phenomenon—the optimizer routes gradients exclusively through the local spelling layers, starving the deep syntax layers of updates. In this paper, we introduce a 3-Tier Continuous Hierarchical State Space Model (HSSM) totaling only 1.51 Million parameters. By introducing a *Cross-Swish Gate* at the output head—which multiplicatively fuses a 1-character local spelling cursor with a dynamically chunked 30-character deep syntactic state—we completely eliminate gradient starvation. The resulting architecture breaks the structural predictive plateau of flat models (improving from 2.60 to 2.34 BPC) and demonstrates robust long-range grammatical reasoning, achieving 57.5% accuracy with a strong positive confidence margin on the BLiMP regular plural subject-verb agreement benchmark.

---

### 1. Introduction
The pursuit of long-context understanding in natural language processing is typically addressed by scaling parameters and expanding attention windows. However, the biological mechanics of language processing suggest a more efficient route: humans do not process language as a flat sequence of equally weighted tokens. Instead, we chunk characters into syllables, syllables into words, and words into abstract grammatical concepts, holding these deep syntactic states in memory while our low-level reflexes handle the mechanics of reading and spelling.

Translating this multi-timescale rhythm into a neural architecture, specifically within the continuous-time framework of State Space Models (SSMs), presents a severe optimization challenge known as the *Spelling vs. Syntax Bottleneck*. 

If a deep neural network is tasked with predicting the next character, the loss function demands exact, high-frequency phonetic data. If the architecture attempts to decouple these tasks by providing the output head with a linear combination of a fast-ticking layer (spelling) and a slow-ticking layer (syntax), the network falls into the *Linear Passthrough Trap*. Because the fast layer provides an easily accessible approximation of the immediate target, the optimizer takes the path of least resistance. It zeroes out the complex grammatical weights, entirely starving the deep layers of gradient signal.

In this work, we propose a radically small, highly efficient 3-Tier HSSM implemented in Elixir/Nx. Rather than relying on brute-force parameter scaling or linear skip connections, we enforce a strict division of labor across temporal horizons using dynamic continuous gating. To ensure the deep syntax engine is actively utilized, we replace the linear output projection with a *Cross-Swish Gate*, a multiplicative fusion mechanism that forces local spelling features to be explicitly gated by deep grammatical context. 

We demonstrate that this 1.51M parameter model natively learns to segment text into ~3-character syllables and ~25-to-30 character syntactic blocks, successfully maintaining un-decayed plural subject-verb agreement across multiple intervening words. Furthermore, we identify and correct a pervasive length-normalization flaw in autoregressive evaluation, revealing the true zero-shot grammatical accuracy of sub-2M parameter systems.

---

### 2. The 3-Tier Architecture: Compressing Time and Space

The fundamental philosophy of the Cross-Swish HSSM is that discrete human language is best modeled as a continuous dynamical system with rhythmic, cascading updates. To achieve this within a highly constrained parameter budget (1.51M parameters), we eschew standard dense attention matrices in favor of Low-Rank Singular Value Decomposition (SVD) factors and dynamic, learned boundary gates.

The architecture processes sequences across three distinct temporal horizons:
1. **Level 1 (Base):** A fast-ticking 1-character cursor.
2. **Level 2 (Mid):** A syllable-level accumulator.
3. **Level 3 (Top):** A deep syntactic memory bank.

#### 2.1 The Low-Rank Physics Engine
At the core of all three tiers is a modified continuous-time State Space Model, structurally analogous to a discretized physics engine. To maintain a large hidden dimension ($D = 512$) without parameter explosion, all state transitions and input projections are factored through a low-rank bottleneck ($R = 64$). 

Instead of a standard dense projection $W \in \mathbb{R}^{D \times D}$, transformations are computed via $W_v(W_u x)$, reducing the parameter cost from $O(D^2)$ to $O(2 \cdot D \cdot R)$. 

The core recurrence at any given layer $L$ updates its continuous memory state $h$ based on the input $x$ and dynamic step size $\Delta$:

$$h_{L}^{(t)} = \tanh\left(e^{-\Delta \cdot |A|} \odot h_{L}^{(t-1)} + \Delta \cdot (W_v (W_u x_t))\right)$$

Where $A$ is a learned diagonal friction matrix that controls the natural decay rate of the memory state, and $\odot$ denotes element-wise multiplication.

#### 2.2 Dynamic Rhythmic Gating
Unlike traditional models that rely on rigid, pre-computed sub-word tokenizers (e.g., Byte-Pair Encoding), this architecture natively learns its own temporal boundaries through unconstrained optimization. Information only flows upward when a lower tier completes a linguistic concept.

This is controlled by a scalar boundary gate $g \in (0, 1)$, generated by projecting the lower layer's state through a learned weight vector and applying a sigmoid activation:

$$g = \sigma(W_{gate} \cdot h_{lower})$$

This gate dictates the continuous interpolation of the upper layer's memory. If $g \approx 0$ (mid-concept), the upper layer ignores the input and retains its previous state. If $g \approx 1$ (concept boundary), the upper layer fully digests the new candidate state $\tilde{h}_{upper}$:

$$h_{upper}^{(t)} = g \odot \tilde{h}_{upper} + (1 - g) \odot h_{upper}^{(t-1)}$$

Simultaneously, a soft-reset mechanism flushes the lower layer's memory, ensuring it is a blank slate ready to accumulate the next sequence of characters:

$$h_{lower}^{(t)} = (1 - g) \odot h_{lower}^{(t)}$$

During training, sparsity penalties are applied to these gates to encourage linguistic chunking. Empirical telemetry confirms that the network naturally settles into highly stable, human-interpretable rhythms: **Gate 1** fires at $\sim 33\%$ (chunking every 3 characters into syllables), and **Gate 2** fires at $\sim 12\%$ (chunking every 8 syllables, or 24–30 characters, into deep syntax).

#### 2.3 The Muon-Adam Hybrid Optimization Strategy
The architectural division between deep temporal memory (matrices) and rhythmic timing (scalars) requires an equally divided optimization strategy. Applying a standard AdamW optimizer globally across the network presents structural risks: Adam's momentum-based updates can slowly degrade the orthogonal properties of the Low-Rank SVD matrices ($W_u$, $W_v$) that drive the state transitions. Conversely, newer matrix-specific optimizers like Muon lack the capacity to natively optimize 1D vectors and scalar biases.

To achieve rapid, stable convergence, we implement a bifurcated optimization step within the Elixir/Nx training loop. 

All 2D matrices (the SVD projections and the Swish Gate weights) are updated using **Muon**. By applying Newton-Schulz iterations to the gradients, Muon implicitly orthogonalizes the updates, ensuring that the low-rank continuous physics engine remains mathematically stable even at high learning rates. 

Simultaneously, all 1D tensors and scalars—specifically the temporal friction diagonals ($A$) and the boundary gate biases ($g_b$)—are routed through standard **Adam**. These parameters dictate the delicate timing of the network's rhythm. They require the precise, momentum-smoothed updates that Adam excels at to prevent the gates from collapsing or wildly oscillating. 

This hybrid approach allows the optimizer to aggressively restructure the grammatical memory space without accidentally destroying the delicate activation thresholds of the boundary gates, directly contributing to the model's rapid descent past the 2.40 BPC threshold.

---

### 3. The Routing Paradoxes

The fundamental challenge in continuous hierarchical modeling is not capturing information, but routing it. When attempting to unify a fast-ticking spelling engine with a slow-ticking grammar engine, architectures inevitably fall into one of three mathematical traps.

#### 3.1 The Frozen Cursor Problem
The most naive approach to hierarchical modeling is strict temporal compression: forcing the entire network to update at a slower rate as information moves upward, and then using only the top-most layer to predict the next token. 

In character-level modeling, this causes immediate phonetic collapse. If the Top Level (Syntax) only updates every 25 characters, the output head is forced to project the exact same continuous state ($h_t$) for 25 consecutive forward passes. Without a fast-ticking contextual "cursor" to tell the output head exactly where it is within a word, the model loses its place, resulting in repeated characters, stuttering, and an inability to correctly spell even basic vocabulary. To spell correctly, the output head requires high-frequency local context.

#### 3.2 The Lazy Shortcut and Gradient Starvation
To resolve the Frozen Cursor Problem, practitioners often introduce a skip connection, routing the fast-ticking Base Level directly to the output head alongside the Top Level. While this restores spelling capability, it introduces a fatal optimization flaw: **Gradient Starvation**.

Neural network optimizers operate on the path of least resistance. In this architecture, the Base Level is a pre-trained, highly efficient phonetic engine, while the Top Level is a randomly initialized, slow-moving matrix. When the optimizer calculates the backward pass, it recognizes that adjusting the direct skip connection from the Base Level yields an immediate, massive reduction in the next-character loss. 

Consequently, the optimizer takes the "Lazy Shortcut." It dumps the vast majority of its gradient updates into the Base Level's projection weights and actively scales the Top Level's gradients toward zero. The deep syntax engine is starved of learning signal, effectively reducing the hierarchical model back into a flat, 1-tier phonetic memorizer. 

#### 3.3 The Linear Passthrough Trap
If aggressive gradient clipping or routing penalties are applied to force the optimizer to train the Top Level, the architecture hits its final structural ceiling: the limitation of linear addition.

In a standard sequence model, the final logits are calculated via a linear projection of the hidden states. If both the fast and slow states are passed to the head, they are simply added together in logit space:

$$y = (W_{fast} \times h_{fast}) + (W_{slow} \times h_{slow}) + b$$

However, grammatical agreement requires conditional, boolean-style logic. For example, if the fast state indicates *"we are at the end of a verb"* and the slow state indicates *"the subject from twenty characters ago is plural,"* the network must suppress the probability of the letter *'s'*. 

Linear addition cannot natively execute this logic. The states merely vote independently. If the fast phonetic state highly predicts an *'s'* based on local n-gram frequencies, it will consistently overpower the slow grammatical state. To execute true conditional gating, the tensors must interact non-linearly before the final vocabulary projection.

---

### 4. The Solution: Multiplicative Swish Fusion

To resolve the paradox of routing—granting the output head access to high-frequency spelling data without triggering gradient starvation—we must abandon linear skip connections. If linear addition allows the optimizer to take a "lazy shortcut," the solution is to mathematically bind the layers together such that they cannot be updated independently.

To achieve this, we introduce the **Cross-Swish Gate**, a multiplicative fusion mechanism that forces the fast-ticking phonetic cursor to be explicitly filtered by the slow-ticking grammatical memory.

#### 4.1 The Mechanism
At the final output head of the network, rather than projecting the concatenated hidden states directly to the vocabulary, we separate the hierarchical states into two distinct pathways: a feature generator and a control valve.

First, we define the local spelling context ($h_{local}$) as the linear addition of the fast-ticking Base and Mid layers. We project this context to generate the raw spelling features ($F_{spell}$):

$$h_{local}=h_{base}+h_{mid}$$
$$F_{spell}=W_{spell} \cdot h_{local}+b_{spell}$$

Simultaneously, we use the slow-ticking Top level (the deep syntactic memory) to generate a control valve. We project the syntactic state, $h_{top}$, and apply the non-linear Swish activation function, where $\text{Swish}(x) = x \cdot \sigma(x)$:

$$V_{syntax}=\text{Swish}(W_{gate} \cdot h_{top}+b_{gate})$$

Finally, we perform the crucial operation: element-wise multiplicative fusion ($\odot$) between the syntactic valve and the local spelling features. This fused state is then normalized and projected to the vocabulary size to produce the final logits:

$$h_{fused}=V_{syntax} \odot F_{spell}$$
$$logits=W_{out} \cdot \text{RMSNorm}(h_{fused})+b_{out}$$

#### 4.2 Curing Gradient Starvation
The Cross-Swish Gate fundamentally alters the backward pass of the network. Because the fusion step relies on multiplication ($V_{syntax} \odot F_{spell}$), the chain rule of calculus dictates that the partial derivative of the spelling features depends directly on the current state of the syntactic valve, and vice versa. 

It is mathematically impossible for the optimizer to reduce the loss by exclusively updating the spelling weights ($W_{spell}$). Any gradient flowing back from the cross-entropy loss must pass *through* the grammatical state. This permanently cures gradient starvation; the optimizer is forced to recognize that the most efficient way to scale, amplify, or suppress the spelling features is to accurately structure the deep memory of the Top level. Empirical telemetry confirms this: upon initializing the Cross-Swish Gate, the maximum gradient norm safely transitions from the output projection weights deep into the Top level's recurrent matrices, remaining highly stable throughout training.

#### 4.3 Executing Boolean Grammatical Logic
Beyond stabilizing the backward pass, multiplicative fusion unlocks the conditional logic required for syntax. While linear addition ($A+B$) only allows independent states to "vote" on an outcome, multiplication allows the Top level to act as a true boolean gate.

If the local spelling context ($h_{local}$) strongly predicts the letter *'s'* based on local n-gram frequencies, but the deep syntactic memory ($h_{top}$) recognizes a plural subject from 25 characters prior, the syntax layer can output a valve vector ($V_{syntax}$) with near-zero values in the corresponding dimensions. Because $0 \cdot 1 = 0$, the syntactic layer successfully suppresses the incorrect phonetic reflex, demonstrating true hierarchical control over the sequence generation.

---

### 5. Experiments & Telemetry

To demonstrate the structural efficiency of the Cross-Swish HSSM, we explicitly avoided large-scale web scrapes. Instead, the model was trained entirely on the **BabyLM strict-small** dataset, a highly constrained corpus of ~10 million words designed to mimic the volume of language a human child is exposed to during early development. Achieving complex syntactic reasoning within this constrained data regime requires profound architectural data efficiency.

#### 5.1 Progressive Layered Pre-Training
Rather than initializing the entire 3-Tier hierarchy from scratch and training it end-to-end—which often leads to temporal confusion as all layers scramble to interpret raw characters simultaneously—we utilized a progressive, layered pre-training curriculum. This mimics human language acquisition, where lower-level phonetic processing crystallizes before higher-level syntactic reasoning fully develops.

The training was segmented into distinct phases:
1. **Phase 1 & 2 (The Frozen Phonetic Foundation):** The base character embeddings and the Level 1 SSM (Physics Engine) were pre-trained on the BabyLM corpus to predict next-character probabilities. Once this layer successfully learned the local statistical manifold of English spelling, its weights were permanently frozen. 
2. **Phase 3 (Syllable Crystallization):** The Level 2 Mid layer and Gate 1 were introduced on top of the frozen foundation. Forced to rely on the frozen base for raw spelling, Level 2 naturally learned to chunk the sequence into syllables, locking its gate frequency at $\sim 33\%$.
3. **Phase 4 (Syntax and Swish Fusion):** Finally, the Top Level and the Cross-Swish Gate were initialized. Because the lower layers were already acting as a highly efficient, frozen syllable engine, the Top Level was insulated from high-frequency phonetic noise. 

This progressive freezing not only drastically reduces the VRAM and compute requirements during the final phases of training but acts as a powerful regularizer. By the time the optimizer began tuning the Cross-Swish Gate, the "infinite" spelling context of the frozen lower layers was mathematically stable, allowing the Top Level to safely expand its memory horizon to 30 characters without suffering from phonetic collapse.

#### 5.2 Hierarchical Stabilization
A core premise of the HSSM is that it should natively discover the optimal temporal boundaries of human language without human-engineered tokenization. During Phase 4 training, the network's boundary gates rapidly stabilized into a fixed, highly interpretable rhythm:
* **Gate 1 (Syllables):** Locked at $\sim 33.8\%$, indicating a consistent memory flush and concept completion every 3 characters.
* **Gate 2 (Syntax):** Oscillated stably between $10.4\%$ and $13.8\%$. 

Mathematically, this confirms the Top level achieved an un-decayed continuous memory horizon of 7 to 10 syllables (approximately 25 to 30 characters). The Cross-Swish Gate allowed this horizon to remain stable without the optimizer attempting to force Gate 2 open to act as a spelling cursor.

#### 5.3 Breaking the Predictive Plateau
Prior iterations of this architecture utilizing linear skip connections hit a rigid structural plateau at 2.60 BPC. The optimizer, suffering from gradient starvation, was unable to utilize the Top level's parameters to improve loss. 

Upon implementing the Cross-Swish Gate, the model effortlessly bypassed the 2.60 barrier, reaching **2.345 BPC** by step 42,950. Notably, this 10% gain in predictive efficiency occurred while the learning rate was still operating at $1.97 \times 10^{-4}$, prior to the final convergence cooldown of the cosine schedule, indicating a profound structural alignment with the dataset rather than mere local-minima settling.

#### 5.4 Zero-Shot Grammatical Reasoning (BLiMP)
To prove the model wasn't just predicting local characters better, but actually understanding long-range grammar, we evaluated it against the `regular_plural_subject_verb_agreement` dataset from the BLiMP (Benchmark of Linguistic Minimal Pairs) suite. 

**The Length Normalization Flaw:** Standard autoregressive evaluation often compares the average loss-per-character between a grammatically correct sentence and an incorrect one. However, in minimal pairs where the incorrect sentence contains an additional character (e.g., "The dogs bark" vs. "The dogs barks"), averaging the loss artificially shrinks the total error of the longer sequence, heavily penalizing the correct model behavior. 

By correcting this evaluation metric to compare the **Total Negative Log-Likelihood** (the pure sum of the sequence loss), the true capability of the HSSM was revealed.

The 1.51M parameter Cross-Swish HSSM achieved **57.5% accuracy** with a positive average confidence margin of 0.2473. By crossing the 50% threshold of random guessing, the network proved that its 30-character syntactic state ($h_{top}$) successfully maintained the concept of a plural subject across intervening spaces and words, utilizing the Swish valve to actively suppress the incorrect phonetic reflex at the end of the sentence.

---

### 6. Conclusion & Future Work
We have demonstrated that the "Spelling vs. Syntax" bottleneck in continuous hierarchical modeling is not a failure of capacity, but a failure of routing. By replacing linear skip connections with a Cross-Swish Gate, we successfully decoupled local statistical memorization from long-range grammatical reasoning. 

The resulting architecture, "Swishful Thinking," proves that massive parameter counts are not strictly necessary for complex linguistic modeling. A 1.51 Million parameter physics engine, operating entirely in continuous time without attention mechanisms, can natively discover the rhythmic structure of human language and execute long-range grammatical logic.

Future work will explore scaling this gating mechanism to larger embedding dimensions, applying it to word-piece tokenizers to extend the grammatical horizon to the paragraph level, and analyzing the precise manifold topology of the isolated syntactic states.
