# microGPT in Rust: 208 Lines, 50x Faster Than Python

*Porting Karpathy's microgpt.py to zero-dependency Rust—same algorithm, comparable line count, 50x the speed.*

---

Andrej Karpathy released [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)—a complete GPT in ~200 lines of pure Python. No PyTorch, no NumPy. Just `math`, `random`, and `os`. The tagline:

> *"This file is the complete algorithm. Everything else is just efficiency."*

So I chased the efficiency. I ported it to Rust: [microgpt-rust.rs](https://gist.github.com/vinodsharma/64f9460d7c9f2ef4dbfe45591c7a6a6e)—208 lines, zero dependencies, 50x faster. Here's what that took.

---

## What microgpt Does

Both versions implement the same thing in roughly the same number of lines:

- **Scalar autograd** — every add, multiply, and exp tracked for backpropagation
- **A GPT-2-style transformer** — embeddings, multi-head attention, RMSNorm, MLP
- **Adam optimizer** — bias correction, linear learning rate decay
- **Training** — 1000 steps on a character-level names dataset (32K names)
- **Inference** — temperature sampling to generate 20 new names

The architecture: 16-dimensional embeddings, 4 attention heads, 1 layer, context length 16, ~4200 parameters. Tiny, but a real transformer.

---

## The Hard Part: Ownership vs. Computation Graphs

Python's `Value` class builds a DAG with shared references. Each node points to its children. Backpropagation walks this graph in reverse.

```python
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
```

This works because Python has garbage collection. Rust doesn't. Shared mutable references require `Rc<RefCell<...>>`, which is verbose, slow, and fights the borrow checker.

The solution: **don't build a graph**. Use a tape.

---

## Tape-Based Autograd

Everything lives in flat arrays. A value is just an index:

```rust
struct V(usize);  // just an index into the tape

struct Tape {
    values: Vec<f64>,   // forward values
    grads:  Vec<f64>,   // backward gradients
    ops:    Vec<Op>,    // what produced each value
    n_params: usize,    // parameters live at indices 0..n_params
}
```

Every operation appends to the tape. `V(42)` means `tape.values[42]`. No references, no lifetimes, no `Rc`.

Backward is trivial—walk the tape in reverse:

```rust
fn backward(&mut self, loss: V) {
    self.grads[loss.0] = 1.0;
    for i in (0..self.values.len()).rev() {
        let g = self.grads[i];
        if g == 0.0 { continue; }
        match self.ops[i] {
            Op::Add(a, b) => { self.grads[a] += g; self.grads[b] += g; }
            Op::Mul(a, b) => {
                self.grads[a] += self.values[b] * g;
                self.grads[b] += self.values[a] * g;
            }
            // ... Pow, Log, Exp, Relu
        }
    }
}
```

No topological sort needed. The tape is already in order by construction.

After each training step, `reset()` truncates back to parameter slots and zeros gradients. Parameters persist; everything else is ephemeral.

---

## What Translated Directly

Most of the model is the same algorithm routing through the tape:

```rust
// Python: x = [t + p for t, p in zip(tok_emb, pos_emb)]
// Rust:
let x: Vec<V> = (0..n_embd)
    .map(|j| tape.add(V(tok_row[j]), V(pos_row[j])))
    .collect();
```

Adam, softmax, RMSNorm, linear layers—all structurally identical.

### What Needed Rethinking

- **RNG:** No built-in `random.gauss()` in Rust. I used a linear congruential generator with Box-Muller transform—keeping the zero-dependency spirit.
- **Parameters:** Python uses a dictionary of nested lists. Rust uses a `Matrix` struct storing indices into the tape.
- **Operators:** Python's `__truediv__` becomes `tape.pow(b, -1.0)` then `tape.mul(a, inv_b)`. Every operator spelled out.

---

## Results

Same dataset, same hyperparameters, same training steps.

| Metric | Python (199 lines) | Rust (208 lines) |
|---|---|---|
| Training time (1000 steps) | ~120s | ~2.5s |
| Per-step time | ~120ms | ~2.5ms |
| Final loss range | ~2.0–2.5 | ~1.7–2.5 |
| Generated names | Plausible | Plausible |

### ~50x faster — comparable line count

Where the speedup comes from:

- **No interpreter overhead:** Python dispatches every `+` and `*` through the object protocol. Rust compiles to native arithmetic.
- **Cache-friendly layout:** The tape is a contiguous `Vec<f64>`. Python's `Value` objects are scattered across the heap.
- **No GC:** Python pauses to trace the computation graph. Rust's tape just truncates.
- **No allocation:** Python creates a new `Value` object per operation. Rust appends an `f64` and an `Op` enum.

---

## What the Port Reveals

Both versions converge to the same loss and generate the same quality of names. The algorithm is identical. The difference is everything below the algorithm—compiled arithmetic vs. interpreter dispatch, contiguous arrays vs. heap-scattered objects, truncating a vector vs. tracing a garbage collector. None of that changes the math. All of it accounts for the 50x.

And this is still just one rung on the ladder. PyTorch adds tensor ops, CUDA, operator fusion, mixed precision, distributed training—each layer building on the last, turning a toy demo into billion-parameter models.

The beauty of microgpt is stripping away all those layers to reveal the core algorithm. The beauty of porting it to Rust is seeing exactly what those layers buy you.

---

## Try It

- [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — Karpathy's original (Python, 199 lines)
- [microgpt-rust.rs](https://gist.github.com/vinodsharma/64f9460d7c9f2ef4dbfe45591c7a6a6e) — Rust port (208 lines, ~50x faster)

```bash
# Rust
curl -O https://gist.githubusercontent.com/vinodsharma/64f9460d7c9f2ef4dbfe45591c7a6a6e/raw/microgpt-rust.rs
rustc -O microgpt-rust.rs -o microgpt && ./microgpt

# Python
curl -O https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/microgpt.py
python3 microgpt.py
```

---

*The algorithm is the idea. The implementation is the trade-off. And the gap between them is where all the interesting engineering lives.*
