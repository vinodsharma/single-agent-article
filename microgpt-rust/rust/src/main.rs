/// microgpt.rs — Port of Karpathy's microgpt.py to Rust, zero dependencies.
use std::fs;
#[derive(Clone, Copy)] struct V(usize);
#[derive(Clone)] enum Op { None, Add(usize, usize), Mul(usize, usize), Pow(usize, f64), Log(usize), Exp(usize), Relu(usize) }
struct Tape { values: Vec<f64>, grads: Vec<f64>, ops: Vec<Op>, n_params: usize }
impl Tape {
    fn new() -> Self { Tape { values: vec![], grads: vec![], ops: vec![], n_params: 0 } }
    fn push(&mut self, v: f64, op: Op) -> V {
        self.values.push(v); self.grads.push(0.0); self.ops.push(op); V(self.values.len() - 1)
    }
    fn param(&mut self, v: f64) -> V { let r = self.push(v, Op::None); self.n_params = self.values.len(); r }
    fn constant(&mut self, v: f64) -> V { self.push(v, Op::None) }
    fn add(&mut self, a: V, b: V) -> V { self.push(self.values[a.0] + self.values[b.0], Op::Add(a.0, b.0)) }
    fn mul(&mut self, a: V, b: V) -> V { self.push(self.values[a.0] * self.values[b.0], Op::Mul(a.0, b.0)) }
    fn pow(&mut self, a: V, e: f64) -> V { self.push(self.values[a.0].powf(e), Op::Pow(a.0, e)) }
    fn log(&mut self, a: V) -> V { self.push(self.values[a.0].ln(), Op::Log(a.0)) }
    fn exp(&mut self, a: V) -> V { self.push(self.values[a.0].exp(), Op::Exp(a.0)) }
    fn relu(&mut self, a: V) -> V { self.push(self.values[a.0].max(0.0), Op::Relu(a.0)) }
    fn backward(&mut self, loss: V) {
        self.grads[loss.0] = 1.0;
        for i in (0..self.values.len()).rev() {
            let g = self.grads[i];
            if g == 0.0 { continue; }
            match self.ops[i].clone() {
                Op::None => {}
                Op::Add(a, b) => { self.grads[a] += g; self.grads[b] += g; }
                Op::Mul(a, b) => { self.grads[a] += self.values[b] * g; self.grads[b] += self.values[a] * g; }
                Op::Pow(a, e) => { self.grads[a] += e * self.values[a].powf(e - 1.0) * g; }
                Op::Log(a) => { self.grads[a] += g / self.values[a]; }
                Op::Exp(a) => { self.grads[a] += self.values[i] * g; }
                Op::Relu(a) => { if self.values[i] > 0.0 { self.grads[a] += g; } }
            }
        }
    }
    fn reset(&mut self) {
        self.values.truncate(self.n_params); self.grads.truncate(self.n_params); self.ops.truncate(self.n_params);
        self.grads.iter_mut().for_each(|g| *g = 0.0);
    }
}

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn f64(&mut self) -> f64 { (self.u64() >> 11) as f64 / ((1u64 << 53) as f64) }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.f64().max(1e-15), self.f64());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() { v.swap(i, (self.u64() as usize) % (i + 1)); }
    }
    fn weighted_choice(&mut self, weights: &[f64]) -> usize {
        let mut r = self.f64() * weights.iter().sum::<f64>();
        for (i, &w) in weights.iter().enumerate() { r -= w; if r <= 0.0 { return i; } }
        weights.len() - 1
    }
}

struct Matrix { data: Vec<usize>, rows: usize, cols: usize }
impl Matrix { fn row(&self, r: usize) -> &[usize] { &self.data[r * self.cols..(r + 1) * self.cols] } }
struct Model {
    n_embd: usize, n_head: usize, block_size: usize, head_dim: usize,
    wte: Matrix, wpe: Matrix, lm_head: Matrix,
    attn_wq: Matrix, attn_wk: Matrix, attn_wv: Matrix, attn_wo: Matrix,
    mlp_fc1: Matrix, mlp_fc2: Matrix,
}
fn init_model(tape: &mut Tape, rng: &mut Rng, vs: usize) -> Model {
    let (ne, bs) = (16, 16);
    let am = |t: &mut Tape, r: &mut Rng, ro: usize, co: usize| -> Matrix {
        Matrix { data: (0..ro * co).map(|_| t.param(r.gauss() * 0.08).0).collect(), rows: ro, cols: co }
    };
    Model { n_embd: ne, n_head: 4, block_size: bs, head_dim: 4,
        wte: am(tape, rng, vs, ne), wpe: am(tape, rng, bs, ne), lm_head: am(tape, rng, vs, ne),
        attn_wq: am(tape, rng, ne, ne), attn_wk: am(tape, rng, ne, ne),
        attn_wv: am(tape, rng, ne, ne), attn_wo: am(tape, rng, ne, ne),
        mlp_fc1: am(tape, rng, 4 * ne, ne), mlp_fc2: am(tape, rng, ne, 4 * ne) }
}

fn linear(tape: &mut Tape, x: &[V], w: &Matrix) -> Vec<V> {
    (0..w.rows).map(|r| {
        let row = w.row(r);
        (1..w.cols).fold(tape.mul(V(row[0]), x[0]), |acc, j| { let p = tape.mul(V(row[j]), x[j]); tape.add(acc, p) })
    }).collect()
}
fn softmax(tape: &mut Tape, logits: &[V]) -> Vec<V> {
    let neg_max = tape.constant(-logits.iter().map(|v| tape.values[v.0]).fold(f64::NEG_INFINITY, f64::max));
    let exps: Vec<V> = logits.iter().map(|v| { let s = tape.add(*v, neg_max); tape.exp(s) }).collect();
    let total = exps[1..].iter().fold(exps[0], |a, &e| tape.add(a, e));
    let inv = tape.pow(total, -1.0);
    exps.iter().map(|e| tape.mul(*e, inv)).collect()
}
fn rmsnorm(tape: &mut Tape, x: &[V]) -> Vec<V> {
    let ms = x[1..].iter().fold(tape.mul(x[0], x[0]), |a, &xi| { let sq = tape.mul(xi, xi); tape.add(a, sq) });
    let c = tape.constant(1.0 / x.len() as f64); let ms = tape.mul(ms, c);
    let e = tape.constant(1e-5); let ms = tape.add(ms, e);
    let scale = tape.pow(ms, -0.5); x.iter().map(|xi| tape.mul(*xi, scale)).collect()
}
fn new_kv() -> (Vec<Vec<V>>, Vec<Vec<V>>) { (vec![], vec![]) }

fn gpt_forward(tape: &mut Tape, model: &Model, token_id: usize, pos_id: usize,
    keys: &mut Vec<Vec<V>>, values: &mut Vec<Vec<V>>) -> Vec<V> {
    let (ne, hd) = (model.n_embd, model.head_dim);
    let (tr, pr) = (model.wte.row(token_id), model.wpe.row(pos_id));
    let mut x: Vec<V> = (0..ne).map(|j| tape.add(V(tr[j]), V(pr[j]))).collect();
    x = rmsnorm(tape, &x);
    // Single transformer layer (n_layer=1)
    let xr = x.clone(); x = rmsnorm(tape, &x);
    let q = linear(tape, &x, &model.attn_wq);
    let k = linear(tape, &x, &model.attn_wk);
    let v = linear(tape, &x, &model.attn_wv);
    keys.push(k); values.push(v.clone());
    let mut xa = Vec::with_capacity(ne);
    let s = tape.constant(1.0 / (hd as f64).sqrt());
    for h in 0..model.n_head {
        let (hs, nt) = (h * hd, keys.len());
        let qh: Vec<V> = q[hs..hs + hd].to_vec();
        let al: Vec<V> = (0..nt).map(|t| {
            let dot = (1..hd).fold(tape.mul(qh[0], keys[t][hs]), |d, j| {
                let p = tape.mul(qh[j], keys[t][hs + j]); tape.add(d, p)
            });
            tape.mul(dot, s)
        }).collect();
        let aw = softmax(tape, &al);
        for j in 0..hd {
            let a = (1..nt).fold(tape.mul(aw[0], values[0][hs + j]), |a, t| {
                let p = tape.mul(aw[t], values[t][hs + j]); tape.add(a, p)
            });
            xa.push(a);
        }
    }
    x = linear(tape, &xa, &model.attn_wo);
    x = x.iter().zip(xr.iter()).map(|(a, b)| tape.add(*a, *b)).collect();
    let xr = x.clone(); x = rmsnorm(tape, &x);
    x = linear(tape, &x, &model.mlp_fc1); x = x.iter().map(|xi| tape.relu(*xi)).collect();
    x = linear(tape, &x, &model.mlp_fc2);
    x = x.iter().zip(xr.iter()).map(|(a, b)| tape.add(*a, *b)).collect();
    linear(tape, &x, &model.lm_head)
}

fn main() {
    let mut rng = Rng(42);
    if !std::path::Path::new("input.txt").exists() {
        assert!(std::process::Command::new("curl").args(["-sL",
            "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt", "-o", "input.txt"])
            .status().unwrap().success(), "download input.txt failed");
    }
    let text = fs::read_to_string("input.txt").expect("cannot read input.txt");
    let mut docs: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut idx: Vec<usize> = (0..docs.len()).collect();
    rng.shuffle(&mut idx);
    docs = idx.iter().map(|&i| docs[i]).collect();
    let chars: Vec<char> = docs.iter().flat_map(|d| d.chars()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let (bos, vocab_size) = (chars.len(), chars.len() + 1);
    let c2i: std::collections::HashMap<char, usize> = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let mut tape = Tape::new();
    let model = init_model(&mut tape, &mut rng, vocab_size);
    let np = tape.n_params;
    println!("num docs: {} | vocab size: {} | num params: {}", docs.len(), vocab_size, np);
    let (lr, b1, b2, eps) = (0.01_f64, 0.85_f64, 0.99_f64, 1e-8_f64);
    let (mut mb, mut vb) = (vec![0.0f64; np], vec![0.0f64; np]);
    let (ns, t0) = (1000, std::time::Instant::now());
    for step in 0..ns {
        tape.reset();
        let doc = docs[step % docs.len()];
        let tok: Vec<usize> = std::iter::once(bos).chain(doc.chars().map(|ch| c2i[&ch])).chain(std::iter::once(bos)).collect();
        let n = model.block_size.min(tok.len() - 1);
        let (mut keys, mut vals) = new_kv();
        let losses: Vec<V> = (0..n).map(|p| {
            let logits = gpt_forward(&mut tape, &model, tok[p], p, &mut keys, &mut vals);
            let probs = softmax(&mut tape, &logits);
            let lp = tape.log(probs[tok[p + 1]]); let neg = tape.constant(-1.0); tape.mul(lp, neg)
        }).collect();
        let sum = losses[1..].iter().fold(losses[0], |a, &l| tape.add(a, l));
        let inv_n = tape.constant(1.0 / n as f64); let loss = tape.mul(sum, inv_n);
        let lv = tape.values[loss.0];
        tape.backward(loss);
        let lrt = lr * (1.0 - step as f64 / ns as f64);
        for i in 0..np {
            let g = tape.grads[i];
            mb[i] = b1 * mb[i] + (1.0 - b1) * g;
            vb[i] = b2 * vb[i] + (1.0 - b2) * g * g;
            let mh = mb[i] / (1.0 - b1.powi(step as i32 + 1));
            let vh = vb[i] / (1.0 - b2.powi(step as i32 + 1));
            tape.values[i] -= lrt * mh / (vh.sqrt() + eps);
        }
        println!("step {:4} / {:4} | loss {:.4}", step + 1, ns, lv);
    }
    let el = t0.elapsed();
    println!("\nTraining time: {:.2}s ({:.2}ms/step)", el.as_secs_f64(), el.as_secs_f64() * 1000.0 / ns as f64);
    println!("\n--- inference (new, hallucinated names) ---");
    for si in 0..20 {
        tape.reset();
        let (mut keys, mut vals) = new_kv();
        let (mut tid, mut out) = (bos, String::new());
        for p in 0..model.block_size {
            let logits = gpt_forward(&mut tape, &model, tid, p, &mut keys, &mut vals);
            let tempered: Vec<V> = logits.iter().map(|l| { let s = tape.constant(2.0); tape.mul(*l, s) }).collect();
            let probs = softmax(&mut tape, &tempered);
            tid = rng.weighted_choice(&probs.iter().map(|p| tape.values[p.0]).collect::<Vec<_>>());
            if tid == bos { break; }
            out.push(chars[tid]);
        }
        println!("sample {:2}: {}", si + 1, out);
    }
}
