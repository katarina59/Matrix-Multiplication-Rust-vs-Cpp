use std::env;
use std::time::Instant;
use rand::Rng;
use rayon::join;

const BASE_CASE: usize = 128;

#[derive(Clone)]
struct Matrix {
    n: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn new(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(n * n);

        for _ in 0..n * n {
            data.push(rng.gen_range(0.0..100.0));
        }

        Self { n, data }
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }
}


fn measure<F, R>(f: F) -> (R, u128)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    (result, start.elapsed().as_millis())
}


fn standard(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.n;
    let mut c = Matrix { n, data: vec![0.0; n * n] };

    for i in 0..n {
        for k in 0..n {
            let aik = a.get(i, k);
            for j in 0..n {
                c.data[i * n + j] += aik * b.get(k, j);
            }
        }
    }

    c
}


fn dc_mul(a: &Matrix, b: &Matrix, ar: usize, ac: usize, br: usize, bc: usize, size: usize) -> Matrix {
    if size <= BASE_CASE {
        let mut sub_a = Matrix { n: size, data: vec![0.0; size * size] };
        let mut sub_b = sub_a.clone();

        for i in 0..size {
            for j in 0..size {
                sub_a.set(i, j, a.get(ar + i, ac + j));
                sub_b.set(i, j, b.get(br + i, bc + j));
            }
        }

        return standard(&sub_a, &sub_b);
    }

    let h = size / 2;

    let (c11, c12) = join(
        || dc_add(a, b, ar, ac, br, bc, ar, ac + h, br + h, bc, h),
        || dc_add(a, b, ar, ac, br, bc + h, ar, ac + h, br + h, bc + h, h),
    );

    let (c21, c22) = join(
        || dc_add(a, b, ar + h, ac, br, bc, ar + h, ac + h, br + h, bc, h),
        || dc_add(a, b, ar + h, ac, br, bc + h, ar + h, ac + h, br + h, bc + h, h),
    );

    assemble(c11, c12, c21, c22)
}

fn dc_add(
    a: &Matrix, b: &Matrix,
    ar1: usize, ac1: usize, br1: usize, bc1: usize,
    ar2: usize, ac2: usize, br2: usize, bc2: usize,
    size: usize,
) -> Matrix {
    let x = dc_mul(a, b, ar1, ac1, br1, bc1, size);
    let y = dc_mul(a, b, ar2, ac2, br2, bc2, size);
    add(&x, &y)
}


fn strassen(a: &Matrix, b: &Matrix) -> Matrix {
    if a.n <= BASE_CASE {
        return standard(a, b);
    }

    let h = a.n / 2;

    let (p1, p2) = join(
        || strassen(&add_view(a, 0, 0, h, h), &add_view(b, 0, 0, h, h)),
        || strassen(&add_view(a, h, 0, h, h), &view(b, 0, 0, h)),
    );

    let (p3, p4) = join(
        || strassen(&view(a, 0, 0, h), &sub_view(b, 0, h, h, h)),
        || strassen(&view(a, h, h, h), &sub_view(b, h, 0, h, h)),
    );

    let (p5, p6) = join(
        || strassen(&add_view(a, 0, 0, 0, h), &view(b, h, h, h)),
        || strassen(&sub_view(a, h, 0, 0, 0), &add_view(b, 0, 0, 0, h)),
    );

    let p7 = strassen(&sub_view(a, 0, h, h, h), &add_view(b, h, 0, h, h));

    let c11 = add(&sub(&add(&p1, &p4), &p5), &p7);
    let c12 = add(&p3, &p5);
    let c21 = add(&p2, &p4);
    let c22 = add(&sub(&add(&p1, &p3), &p2), &p6);

    assemble(c11, c12, c21, c22)
}


fn view(m: &Matrix, r: usize, c: usize, size: usize) -> Matrix {
    let mut v = Matrix { n: size, data: vec![0.0; size * size] };
    for i in 0..size {
        for j in 0..size {
            v.set(i, j, m.get(r + i, c + j));
        }
    }
    v
}

fn add_view(m: &Matrix, r1: usize, c1: usize, r2: usize, c2: usize) -> Matrix {
    let size = m.n / 2;
    let mut v = Matrix { n: size, data: vec![0.0; size * size] };
    for i in 0..size {
        for j in 0..size {
            v.set(i, j, m.get(r1 + i, c1 + j) + m.get(r2 + i, c2 + j));
        }
    }
    v
}

fn sub_view(m: &Matrix, r1: usize, c1: usize, r2: usize, c2: usize) -> Matrix {
    let size = m.n / 2;
    let mut v = Matrix { n: size, data: vec![0.0; size * size] };
    for i in 0..size {
        for j in 0..size {
            v.set(i, j, m.get(r1 + i, c1 + j) - m.get(r2 + i, c2 + j));
        }
    }
    v
}

fn add(x: &Matrix, y: &Matrix) -> Matrix {
    let mut r = x.clone();
    for i in 0..r.data.len() {
        r.data[i] += y.data[i];
    }
    r
}

fn sub(x: &Matrix, y: &Matrix) -> Matrix {
    let mut r = x.clone();
    for i in 0..r.data.len() {
        r.data[i] -= y.data[i];
    }
    r
}

fn assemble(c11: Matrix, c12: Matrix, c21: Matrix, c22: Matrix) -> Matrix {
    let n = c11.n * 2;
    let mut r = Matrix { n, data: vec![0.0; n * n] };
    let h = c11.n;

    for i in 0..h {
        for j in 0..h {
            r.set(i, j, c11.get(i, j));
            r.set(i, j + h, c12.get(i, j));
            r.set(i + h, j, c21.get(i, j));
            r.set(i + h, j + h, c22.get(i, j));
        }
    }
    r
}


fn approx_equal(a: &Matrix, b: &Matrix) -> bool {
    const EPS: f64 = 1e-8;
    a.data.iter().zip(&b.data).all(|(x, y)| {
        let diff = (x - y).abs();
        diff <= EPS * x.abs().max(y.abs()).max(1.0)
    })
}


fn main() {
    let mut args = env::args().skip(1);
    let n: usize = args.next().expect("missing size").parse().expect("n must be integer");
    let test = matches!(args.next().as_deref(), Some("test"));

    let a = Matrix::new(n);
    let b = Matrix::new(n);

    let (r_std, t_std) = measure(|| standard(&a, &b));
    let (r_dc, t_dc) = measure(|| dc_mul(&a, &b, 0, 0, 0, 0, n));
    let (r_str, t_str) = measure(|| strassen(&a, &b));

    if test {
        println!("{} {} {}", t_std, t_dc, t_str);
        return;
    }

    println!("Correctness:");
    println!("DC: {}", approx_equal(&r_std, &r_dc));
    println!("Strassen: {}", approx_equal(&r_std, &r_str));

    println!("\nTimes (ms):");
    println!("Standard: {}", t_std);
    println!("DC: {}", t_dc);
    println!("Strassen: {}", t_str);
}
