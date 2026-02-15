# Porting Karpathy's microGPT from Python to Rust: Everything Else Is Just Efficiency

*What happens when you take a 200-line pure-Python GPT and rewrite it in Rust? A deep look at tape-based autograd, ownership trade-offs, and what "efficiency" really means.*

---

Andrej Karpathy recently released [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)—a complete GPT implementation in ~200 lines of pure Python with zero dependencies. No PyTorch, no NumPy. Just `math`, `random`, and `os`. The tagline:

> *"The most atomic way to train and inference a GPT in pure, dependency-free Python. This file is the complete algorithm. Everything else is just efficiency."*

That tagline is an invitation. If everything else is just efficiency, what happens when you chase that efficiency as far as it goes? What does the "everything else" actually look like?

I ported microgpt.py to Rust. Here's what I learned.

---

## What microgpt.py Actually Does

In ~200 lines, Karpathy's code implements:

- **Scalar autograd** — A `Value` class that builds a computation graph, tracking every add, multiply, and exp so it can backpropagate gradients
- **A GPT-2-style transformer** — Token and position embeddings, multi-head attention with KV cache, RMSNorm, and a feed-forward MLP
- **Adam optimizer** — With bias correction and linear learning rate decay
- **Training loop** — 1000 steps over a character-level names dataset
- **Inference** — Temperature-controlled sampling to generate 20 new names

The architecture: 16-dimensional embeddings, 4 attention heads, 1 layer, context length of 16, ~4200 parameters. Tiny, but a real transformer.

---

## The Porting Challenge: Ownership vs. Computation Graphs

Python's `Value` class is elegant. Each value holds references to its children, forming a DAG (directed acyclic graph). Backpropagation walks this graph in reverse topological order.

```python
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
```

This works in Python because of reference counting and garbage collection. Multiple values can point to the same child. The graph can be arbitrarily tangled.

In Rust, this is a problem. Rust's ownership model demands that every value has exactly one owner. Shared mutable references require `Rc<RefCell<...>>`, which is verbose, slow, and fights the borrow checker at every turn.

The solution: **don't build a graph at all**. Use a tape.

---

## Tape-Based Autograd

Instead of each value holding references to its children, we store everything in flat arrays:

```rust
struct V(usize);  // just an index

enum Op {
    None,
    Add(usize, usize),
    Mul(usize, usize),
    Pow(usize, f64),
    Log(usize),
    Exp(usize),
    Relu(usize),
}

struct Tape {
    values: Vec<f64>,
    grads:  Vec<f64>,
    ops:    Vec<Op>,
    n_params: usize,
}
```

Every operation appends to the tape. `V(42)` doesn't hold a value—it's an index into `tape.values[42]`. No references, no lifetimes, no `Rc`.

The backward pass is trivial: walk the tape in reverse, accumulate gradients.

```rust
fn backward(&mut self, loss: V) {
    self.grads[loss.0] = 1.0;
    for i in (0..self.values.len()).rev() {
        let g = self.grads[i];
        if g == 0.0 { continue; }
        match self.ops[i] {
            Op::Add(a, b) => {
                self.grads[a] += g;
                self.grads[b] += g;
            }
            Op::Mul(a, b) => {
                self.grads[a] += self.values[b] * g;
                self.grads[b] += self.values[a] * g;
            }
            // ... other ops
        }
    }
}
```

No topological sort needed. The tape is already in topological order by construction—an operation can only reference earlier entries.

After each training step, the tape resets: truncate back to the parameter slots, zero the gradients. Parameters live at indices `0..n_params`; everything else is ephemeral.

---

## What Mapped Cleanly

Most of the model translated directly. The GPT forward pass—embeddings, attention, MLP—is the same algorithm, just routing through the tape instead of building `Value` objects:

```rust
// Python: x = [t + p for t, p in zip(tok_emb, pos_emb)]
// Rust:
let x: Vec<V> = (0..n_embd)
    .map(|j| tape.add(V(tok_row[j]), V(pos_row[j])))
    .collect();
```

The Adam optimizer is almost line-for-line. Softmax, RMSNorm, and linear layers are structurally identical.

### What Needed Rethinking

- **Random number generation:** Python has `random.gauss()`. Rust has no built-in RNG (without the `rand` crate). I implemented xoshiro256** with Box-Muller transform—20 lines replacing one function call.
- **Parameter initialization:** Python uses a dictionary of nested lists. Rust uses a `Matrix` struct that stores parameter indices into the tape, with row-major access.
- **The division operator:** Python's `__truediv__` becomes `tape.pow(b, -1.0)` followed by `tape.mul(a, b_inv)`. Every convenience operator needs to be explicit.

---

## Performance Results

Same dataset (32K names), same hyperparameters, same number of training steps.

| Metric | Python | Rust |
|---|---|---|
| Training time (1000 steps) | ~120s | ~2.6s |
| Per-step time | ~120ms | ~2.6ms |
| Final loss range | ~2.0–2.5 | ~1.9–2.4 |
| Generated names | Plausible | Plausible |

### ~50x faster

The speedup comes from multiple factors working together:

- **No interpreter overhead:** Python dispatches every `+` and `*` through the object protocol. Rust compiles to native arithmetic.
- **Cache-friendly memory layout:** The tape stores all values in a contiguous `Vec<f64>`. Python's `Value` objects are scattered across the heap.
- **No garbage collection:** Python's GC pauses to trace the computation graph. Rust's tape just truncates.
- **No object creation:** Python allocates a new `Value` object for every operation. Rust appends an `f64` and an `Op` enum.

---

## What "Everything Else Is Just Efficiency" Really Means

Karpathy's tagline is precisely correct, but the implications run deeper than they first appear.

The Python version and the Rust version implement the *same algorithm*. The same math, the same architecture, the same optimizer. They converge to the same loss and generate the same quality of output.

The difference is purely in the *representation* of that algorithm:

- Python represents the computation graph as a network of heap-allocated objects with reference-counted pointers
- Rust represents it as indices into contiguous arrays

Both are correct. But one is 50x faster. That's the "everything else."

And this is just one step on the ladder. Real frameworks like PyTorch add tensor operations (BLAS, CUDA), operator fusion, mixed precision, distributed training. Each step is "just efficiency"—but those steps are what make training billion-parameter models possible instead of a toy demo.

The beauty of microgpt is that it strips away all those layers to reveal the core algorithm. The beauty of porting it is seeing exactly what those layers buy you.

---

## Try It Yourself

The complete code is on GitHub:

- [Karpathy's original microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [Rust port + benchmark scripts](https://github.com/vinodsharma/social-media-update/tree/microgpt-rust-port/microgpt-rust)

```bash
# Run the Rust version
cd microgpt-rust/rust
cargo run --release

# Run the Python version
cd microgpt-rust/python
python3 microgpt.py

# Run benchmarks
cd microgpt-rust/benchmark
./run_benchmarks.sh
```

---

*The algorithm is the idea. The implementation is the trade-off. And the gap between them is where all the interesting engineering lives.*
