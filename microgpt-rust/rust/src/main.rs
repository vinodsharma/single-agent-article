/// microgpt.rs — A complete GPT trained and inferenced in pure Rust, zero dependencies.
/// Port of Karpathy's microgpt.py. Everything else is just efficiency.

use std::fs;

// ---------------------------------------------------------------------------
// Tape-based autograd
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct V(usize);

#[derive(Clone)]
enum Op {
    None,
    Add(usize, usize),
    Mul(usize, usize),
    Pow(usize, f64),    // base idx, exponent constant
    Log(usize),
    Exp(usize),
    Relu(usize),
}

struct Tape {
    values: Vec<f64>,
    grads: Vec<f64>,
    ops: Vec<Op>,
    n_params: usize,
}

impl Tape {
    fn new() -> Self {
        Tape { values: Vec::new(), grads: Vec::new(), ops: Vec::new(), n_params: 0 }
    }

    fn param(&mut self, val: f64) -> V {
        let i = self.values.len();
        self.values.push(val);
        self.grads.push(0.0);
        self.ops.push(Op::None);
        self.n_params = self.values.len();
        V(i)
    }

    fn constant(&mut self, val: f64) -> V {
        let i = self.values.len();
        self.values.push(val);
        self.grads.push(0.0);
        self.ops.push(Op::None);
        V(i)
    }

    fn add(&mut self, a: V, b: V) -> V {
        let val = self.values[a.0] + self.values[b.0];
        let i = self.values.len();
        self.values.push(val);
        self.grads.push(0.0);
        self.ops.push(Op::Add(a.0, b.0));
        V(i)
    }

    fn mul(&mut self, a: V, b: V) -> V {
        let val = self.values[a.0] * self.values[b.0];
        let i = self.values.len();
        self.values.push(val);
        self.grads.push(0.0);
        self.ops.push(Op::Mul(a.0, b.0));
        V(i)
    }

    fn pow(&mut self, a: V, exp: f64) -> V {
        let val = self.values[a.0].powf(exp);
        let i = self.values.len();
        self.values.push(val);
        self.grads.push(0.0);
        self.ops.push(Op::Pow(a.0, exp));
        V(i)
    }

    fn log(&mut self, a: V) -> V {
        let val = self.values[a.0].ln();
        let i = self.values.len();
        self.values.push(val);
        self.grads.push(0.0);
        self.ops.push(Op::Log(a.0));
        V(i)
    }

    fn exp(&mut self, a: V) -> V {
        let val = self.values[a.0].exp();
        let i = self.values.len();
        self.values.push(val);
        self.grads.push(0.0);
        self.ops.push(Op::Exp(a.0));
        V(i)
    }

    fn relu(&mut self, a: V) -> V {
        let val = if self.values[a.0] > 0.0 { self.values[a.0] } else { 0.0 };
        let i = self.values.len();
        self.values.push(val);
        self.grads.push(0.0);
        self.ops.push(Op::Relu(a.0));
        V(i)
    }

    fn sub(&mut self, a: V, b: V) -> V {
        let neg_one = self.constant(-1.0);
        let neg_b = self.mul(b, neg_one);
        self.add(a, neg_b)
    }

    fn div(&mut self, a: V, b: V) -> V {
        let b_inv = self.pow(b, -1.0);
        self.mul(a, b_inv)
    }

    fn neg(&mut self, a: V) -> V {
        let neg_one = self.constant(-1.0);
        self.mul(a, neg_one)
    }

    fn scale(&mut self, a: V, s: f64) -> V {
        let sv = self.constant(s);
        self.mul(a, sv)
    }

    fn backward(&mut self, loss: V) {
        self.grads[loss.0] = 1.0;
        for i in (0..self.values.len()).rev() {
            let g = self.grads[i];
            if g == 0.0 { continue; }
            match self.ops[i].clone() {
                Op::None => {}
                Op::Add(a, b) => {
                    self.grads[a] += g;
                    self.grads[b] += g;
                }
                Op::Mul(a, b) => {
                    self.grads[a] += self.values[b] * g;
                    self.grads[b] += self.values[a] * g;
                }
                Op::Pow(a, exp) => {
                    self.grads[a] += exp * self.values[a].powf(exp - 1.0) * g;
                }
                Op::Log(a) => {
                    self.grads[a] += g / self.values[a];
                }
                Op::Exp(a) => {
                    self.grads[a] += self.values[i] * g;
                }
                Op::Relu(a) => {
                    if self.values[i] > 0.0 {
                        self.grads[a] += g;
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        self.values.truncate(self.n_params);
        self.grads.truncate(self.n_params);
        self.ops.truncate(self.n_params);
        for g in self.grads.iter_mut() {
            *g = 0.0;
        }
    }

    fn val(&self, v: V) -> f64 {
        self.values[v.0]
    }

    fn grad(&self, idx: usize) -> f64 {
        self.grads[idx]
    }

    fn set_param(&mut self, idx: usize, val: f64) {
        self.values[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Inline PRNG — xoshiro256** (no dependencies)
// ---------------------------------------------------------------------------

struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64 to seed xoshiro state
        let mut z = seed;
        let mut s = [0u64; 4];
        for si in s.iter_mut() {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut r = z;
            r = (r ^ (r >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            r = (r ^ (r >> 27)).wrapping_mul(0x94d049bb133111eb);
            *si = r ^ (r >> 31);
        }
        Rng { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn gauss(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            v.swap(i, j);
        }
    }

    fn weighted_choice(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut r = self.next_f64() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

// ---------------------------------------------------------------------------
// Model — param indices stored as ranges/matrices
// ---------------------------------------------------------------------------

struct Matrix {
    data: Vec<usize>, // flattened param indices, row-major
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn row(&self, r: usize) -> &[usize] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

struct Model {
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    block_size: usize,
    head_dim: usize,
    #[allow(dead_code)]
    vocab_size: usize,
    wte: Matrix,
    wpe: Matrix,
    lm_head: Matrix,
    attn_wq: Vec<Matrix>,
    attn_wk: Vec<Matrix>,
    attn_wv: Vec<Matrix>,
    attn_wo: Vec<Matrix>,
    mlp_fc1: Vec<Matrix>,
    mlp_fc2: Vec<Matrix>,
}

fn alloc_matrix(tape: &mut Tape, rng: &mut Rng, rows: usize, cols: usize, std: f64) -> Matrix {
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows {
        for _ in 0..cols {
            let v = tape.param(rng.gauss() * std);
            data.push(v.0);
        }
    }
    Matrix { data, rows, cols }
}

fn init_model(tape: &mut Tape, rng: &mut Rng, vocab_size: usize) -> Model {
    let n_embd = 16;
    let n_head = 4;
    let n_layer = 1;
    let block_size = 16;
    let head_dim = n_embd / n_head;
    let std = 0.08;

    let wte = alloc_matrix(tape, rng, vocab_size, n_embd, std);
    let wpe = alloc_matrix(tape, rng, block_size, n_embd, std);
    let lm_head = alloc_matrix(tape, rng, vocab_size, n_embd, std);

    let mut attn_wq = Vec::new();
    let mut attn_wk = Vec::new();
    let mut attn_wv = Vec::new();
    let mut attn_wo = Vec::new();
    let mut mlp_fc1 = Vec::new();
    let mut mlp_fc2 = Vec::new();

    for _ in 0..n_layer {
        attn_wq.push(alloc_matrix(tape, rng, n_embd, n_embd, std));
        attn_wk.push(alloc_matrix(tape, rng, n_embd, n_embd, std));
        attn_wv.push(alloc_matrix(tape, rng, n_embd, n_embd, std));
        attn_wo.push(alloc_matrix(tape, rng, n_embd, n_embd, std));
        mlp_fc1.push(alloc_matrix(tape, rng, 4 * n_embd, n_embd, std));
        mlp_fc2.push(alloc_matrix(tape, rng, n_embd, 4 * n_embd, std));
    }

    Model {
        n_embd, n_head, n_layer, block_size, head_dim, vocab_size,
        wte, wpe, lm_head,
        attn_wq, attn_wk, attn_wv, attn_wo,
        mlp_fc1, mlp_fc2,
    }
}

// ---------------------------------------------------------------------------
// Forward pass functions
// ---------------------------------------------------------------------------

fn linear(tape: &mut Tape, x: &[V], w: &Matrix) -> Vec<V> {
    let mut out = Vec::with_capacity(w.rows);
    for r in 0..w.rows {
        let row = w.row(r);
        let mut acc = tape.mul(V(row[0]), x[0]);
        for j in 1..w.cols {
            let prod = tape.mul(V(row[j]), x[j]);
            acc = tape.add(acc, prod);
        }
        out.push(acc);
    }
    out
}

fn softmax(tape: &mut Tape, logits: &[V]) -> Vec<V> {
    let max_val = logits.iter().map(|v| tape.val(*v)).fold(f64::NEG_INFINITY, f64::max);
    let max_v = tape.constant(max_val);
    let exps: Vec<V> = logits.iter().map(|v| {
        let shifted = tape.sub(*v, max_v);
        tape.exp(shifted)
    }).collect();
    let mut total = exps[0];
    for e in &exps[1..] {
        total = tape.add(total, *e);
    }
    exps.iter().map(|e| tape.div(*e, total)).collect()
}

fn rmsnorm(tape: &mut Tape, x: &[V]) -> Vec<V> {
    let n = x.len();
    let mut ms = tape.mul(x[0], x[0]);
    for xi in &x[1..] {
        let sq = tape.mul(*xi, *xi);
        ms = tape.add(ms, sq);
    }
    let inv_n = tape.constant(1.0 / n as f64);
    ms = tape.mul(ms, inv_n);
    let eps = tape.constant(1e-5);
    ms = tape.add(ms, eps);
    let scale = tape.pow(ms, -0.5);
    x.iter().map(|xi| tape.mul(*xi, scale)).collect()
}

fn gpt_forward(
    tape: &mut Tape,
    model: &Model,
    token_id: usize,
    pos_id: usize,
    keys: &mut Vec<Vec<Vec<V>>>,   // [layer][time][embd]
    values: &mut Vec<Vec<Vec<V>>>, // [layer][time][embd]
) -> Vec<V> {
    let n_embd = model.n_embd;

    // Token + position embedding
    let tok_row = model.wte.row(token_id);
    let pos_row = model.wpe.row(pos_id);
    let mut x: Vec<V> = (0..n_embd).map(|j| tape.add(V(tok_row[j]), V(pos_row[j]))).collect();
    x = rmsnorm(tape, &x);

    for li in 0..model.n_layer {
        // 1) Multi-head attention
        let x_residual = x.clone();
        x = rmsnorm(tape, &x);
        let q = linear(tape, &x, &model.attn_wq[li]);
        let k = linear(tape, &x, &model.attn_wk[li]);
        let v = linear(tape, &x, &model.attn_wv[li]);
        keys[li].push(k);
        values[li].push(v.clone());

        let mut x_attn = Vec::with_capacity(n_embd);
        let inv_sqrt_hd = 1.0 / (model.head_dim as f64).sqrt();

        for h in 0..model.n_head {
            let hs = h * model.head_dim;
            let he = hs + model.head_dim;

            let q_h: Vec<V> = q[hs..he].to_vec();
            let n_time = keys[li].len();

            // Attention logits
            let mut attn_logits = Vec::with_capacity(n_time);
            for t in 0..n_time {
                let k_h: &[V] = &keys[li][t][hs..he];
                let mut dot = tape.mul(q_h[0], k_h[0]);
                for j in 1..model.head_dim {
                    let prod = tape.mul(q_h[j], k_h[j]);
                    dot = tape.add(dot, prod);
                }
                let scaled = tape.scale(dot, inv_sqrt_hd);
                attn_logits.push(scaled);
            }

            let attn_weights = softmax(tape, &attn_logits);

            // Weighted sum of values
            for j in 0..model.head_dim {
                let mut acc = tape.mul(attn_weights[0], values[li][0][hs + j]);
                for t in 1..n_time {
                    let prod = tape.mul(attn_weights[t], values[li][t][hs + j]);
                    acc = tape.add(acc, prod);
                }
                x_attn.push(acc);
            }
        }

        x = linear(tape, &x_attn, &model.attn_wo[li]);
        x = x.iter().zip(x_residual.iter()).map(|(a, b)| tape.add(*a, *b)).collect();

        // 2) MLP block
        let x_residual = x.clone();
        x = rmsnorm(tape, &x);
        x = linear(tape, &x, &model.mlp_fc1[li]);
        x = x.iter().map(|xi| tape.relu(*xi)).collect();
        x = linear(tape, &x, &model.mlp_fc2[li]);
        x = x.iter().zip(x_residual.iter()).map(|(a, b)| tape.add(*a, *b)).collect();
    }

    linear(tape, &x, &model.lm_head)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let mut rng = Rng::new(42);

    // Load dataset
    if !std::path::Path::new("input.txt").exists() {
        eprintln!("Downloading input.txt...");
        let status = std::process::Command::new("curl")
            .args(["-sL", "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt", "-o", "input.txt"])
            .status()
            .expect("failed to run curl");
        if !status.success() {
            panic!("failed to download input.txt");
        }
    }

    let text = fs::read_to_string("input.txt").expect("cannot read input.txt");
    let mut docs: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    // Shuffle docs
    let mut idx: Vec<usize> = (0..docs.len()).collect();
    rng.shuffle(&mut idx);
    let shuffled: Vec<&str> = idx.iter().map(|&i| docs[i]).collect();
    docs = shuffled;
    println!("num docs: {}", docs.len());

    // Tokenizer
    let mut chars_set: Vec<char> = docs.iter().flat_map(|d| d.chars()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    chars_set.sort();
    let bos = chars_set.len();
    let vocab_size = chars_set.len() + 1;
    println!("vocab size: {vocab_size}");

    let char_to_idx: std::collections::HashMap<char, usize> = chars_set.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    // Initialize tape and model
    let mut tape = Tape::new();
    let model = init_model(&mut tape, &mut rng, vocab_size);
    let n_params = tape.n_params;
    println!("num params: {n_params}");

    // Adam optimizer state
    let learning_rate = 0.01_f64;
    let beta1 = 0.85_f64;
    let beta2 = 0.99_f64;
    let eps_adam = 1e-8_f64;
    let mut m_buf = vec![0.0_f64; n_params];
    let mut v_buf = vec![0.0_f64; n_params];

    // Training
    let num_steps = 1000;
    let t_start = std::time::Instant::now();

    for step in 0..num_steps {
        tape.reset();

        let doc = docs[step % docs.len()];
        let mut tokens: Vec<usize> = Vec::new();
        tokens.push(bos);
        for ch in doc.chars() {
            tokens.push(char_to_idx[&ch]);
        }
        tokens.push(bos);
        let n = std::cmp::min(model.block_size, tokens.len() - 1);

        // Forward pass
        let mut keys: Vec<Vec<Vec<V>>> = (0..model.n_layer).map(|_| Vec::new()).collect();
        let mut vals: Vec<Vec<Vec<V>>> = (0..model.n_layer).map(|_| Vec::new()).collect();
        let mut losses: Vec<V> = Vec::new();

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = gpt_forward(&mut tape, &model, token_id, pos_id, &mut keys, &mut vals);
            let probs = softmax(&mut tape, &logits);
            let log_p = tape.log(probs[target_id]);
            let neg_log_p = tape.neg(log_p);
            losses.push(neg_log_p);
        }

        // Average loss
        let mut loss = losses[0];
        for l in &losses[1..] {
            loss = tape.add(loss, *l);
        }
        let inv_n = tape.constant(1.0 / n as f64);
        loss = tape.mul(loss, inv_n);

        let loss_val = tape.val(loss);

        // Backward pass
        tape.backward(loss);

        // Adam update
        let lr_t = learning_rate * (1.0 - step as f64 / num_steps as f64);
        for i in 0..n_params {
            let g = tape.grad(i);
            m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g;
            v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g * g;
            let m_hat = m_buf[i] / (1.0 - beta1.powi(step as i32 + 1));
            let v_hat = v_buf[i] / (1.0 - beta2.powi(step as i32 + 1));
            let current = tape.values[i];
            tape.set_param(i, current - lr_t * m_hat / (v_hat.sqrt() + eps_adam));
        }

        println!("step {:4} / {:4} | loss {:.4}", step + 1, num_steps, loss_val);
    }

    let elapsed = t_start.elapsed();
    println!("\nTraining time: {:.2}s ({:.2}ms/step)", elapsed.as_secs_f64(), elapsed.as_secs_f64() * 1000.0 / num_steps as f64);

    // Inference
    let temperature = 0.5_f64;
    println!("\n--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        tape.reset();
        let mut keys: Vec<Vec<Vec<V>>> = (0..model.n_layer).map(|_| Vec::new()).collect();
        let mut vals: Vec<Vec<Vec<V>>> = (0..model.n_layer).map(|_| Vec::new()).collect();
        let mut token_id = bos;
        let mut sample = String::new();

        for pos_id in 0..model.block_size {
            let logits = gpt_forward(&mut tape, &model, token_id, pos_id, &mut keys, &mut vals);

            // Apply temperature
            let tempered: Vec<V> = logits.iter().map(|l| tape.scale(*l, 1.0 / temperature)).collect();
            let probs = softmax(&mut tape, &tempered);
            let weights: Vec<f64> = probs.iter().map(|p| tape.val(*p)).collect();

            token_id = rng.weighted_choice(&weights);
            if token_id == bos {
                break;
            }
            sample.push(chars_set[token_id]);

            // Reset tape but preserve params — we don't need gradients for inference
            // Keep the KV cache alive by not resetting; for inference the tape grows per token
            // which is fine since sequences are short
        }

        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
}
