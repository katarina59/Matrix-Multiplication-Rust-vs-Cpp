use std::time::Instant;
use std::env;
use rand::Rng;
use rayon::prelude::*;

const BASE_CASE: usize = 128;
fn main() {
    let mut args = env::args().skip(1);

    let n: usize = match args.next() {
        Some(value) => match value.parse() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("Error: matrix dimension must be an integer");
                return;
            }
        },
        None => {
            println!("Usage:");
            println!("  program <n> [max_parallel_depth] [test]");
            return;
        }
    };
    
    let max_parallel_depth: usize = match args.next() {
        Some(value) => value.parse().unwrap_or(3),
        None => 3,
    };
    
    let test_mode: bool = matches!(args.next().as_deref(), Some("test"));

    let matrices = (
        create_random_matrix(n, n),
        create_random_matrix(n, n),
    );
    
    let (a, b) = (&matrices.0, &matrices.1);
    
    let (standard_res, t_standard) = {
        let start = Instant::now();
        let result = standard_implementation(a, b);
        (result, start.elapsed())
    };
    
    let (dc_res, t_dc) = {
        let start = Instant::now();
        let result = divide_and_conquer_implementation(a, b, 0, max_parallel_depth);
        (result, start.elapsed())
    };
    
    let (strassen_res, t_strassen) = {
        let start = Instant::now();
        let result = strassen_implementation(a, b, 0, max_parallel_depth);
        (result, start.elapsed())
    };
    
    if test_mode {
        println!(
            "{} {} {}",
            t_standard.as_millis(),
            t_dc.as_millis(),
            t_strassen.as_millis()
        );
        return;
    }
    
    let dc_ok = compare_matrices(&standard_res, &dc_res);
    let strassen_ok = compare_matrices(&standard_res, &strassen_res);
    
    println!(
        "Correctness check:\n\
         - Divide & Conquer: {}\n\
         - Strassen: {}\n",
        if dc_ok { "OK" } else { "EROOR" },
        if strassen_ok { "OK" } else { "ERROR" }
    );
    
    println!("Matrix size: {} × {}\n", n, n);
    
    println!(
        "Execution times (ms):\n\
         - Standard: {}\n\
         - Divide & Conquer: {}\n\
         - Strassen: {}\n",
        t_standard.as_millis(),
        t_dc.as_millis(),
        t_strassen.as_millis()
    );
    

}

fn standard_implementation(
    a: &Vec<Vec<f64>>, 
    b: &Vec<Vec<f64>>
) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            for j in 0..n {
                result[i][j] += aik * b[k][j];
            }
        }
    }

    result
}


fn divide_and_conquer_implementation(
    a: &Vec<Vec<f64>>,
    b: &Vec<Vec<f64>>,
    level: usize,
    max_parallel_level: usize,
) -> Vec<Vec<f64>> {
    let size = a.len();

    if size <= BASE_CASE {
        return standard_implementation(a, b);
    }

    let half = size / 2;
    let mut result = vec![vec![0.0; size]; size];

    let (a11, a12, a21, a22) = (
        extract_block(a, 0, 0, half),
        extract_block(a, 0, half, half),
        extract_block(a, half, 0, half),
        extract_block(a, half, half, half),
    );

    let (b11, b12, b21, b22) = (
        extract_block(b, 0, 0, half),
        extract_block(b, 0, half, half),
        extract_block(b, half, 0, half),
        extract_block(b, half, half, half),
    );

    let compute = |x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>| {
        divide_and_conquer_implementation(x, y, level + 1, max_parallel_level)
    };

    let (c11a, c11b, c12a, c12b, c21a, c21b, c22a, c22b) =
        if level < max_parallel_level {
            let (c11a, c11b) = rayon::join(|| compute(&a11, &b11), || compute(&a12, &b21));
            let (c12a, c12b) = rayon::join(|| compute(&a11, &b12), || compute(&a12, &b22));
            let (c21a, c21b) = rayon::join(|| compute(&a21, &b11), || compute(&a22, &b21));
            let (c22a, c22b) = rayon::join(|| compute(&a21, &b12), || compute(&a22, &b22));

            (c11a, c11b, c12a, c12b, c21a, c21b, c22a, c22b)
        } else {
            (
                compute(&a11, &b11),
                compute(&a12, &b21),
                compute(&a11, &b12),
                compute(&a12, &b22),
                compute(&a21, &b11),
                compute(&a22, &b21),
                compute(&a21, &b12),
                compute(&a22, &b22),
            )
        };


    let c11 = add_blocks(&c11a, &c11b);
    let c12 = add_blocks(&c12a, &c12b);
    let c21 = add_blocks(&c21a, &c21b);
    let c22 = add_blocks(&c22a, &c22b);

    insert_block(&mut result, &c11, 0, 0);
    insert_block(&mut result, &c12, 0, half);
    insert_block(&mut result, &c21, half, 0);
    insert_block(&mut result, &c22, half, half);

    result
}


fn strassen_implementation(
    a: &Vec<Vec<f64>>,
    b: &Vec<Vec<f64>>,
    depth: usize,
    max_parallel_depth: usize,
) -> Vec<Vec<f64>> {
    let size = a.len();

    if size <= BASE_CASE {
        return standard_implementation(a, b);
    }

    let half = size / 2;
    let mut result = vec![vec![0.0; size]; size];

    let (a11, a12, a21, a22) = (
        extract_block(a, 0, 0, half),
        extract_block(a, 0, half, half),
        extract_block(a, half, 0, half),
        extract_block(a, half, half, half),
    );

    let (b11, b12, b21, b22) = (
        extract_block(b, 0, 0, half),
        extract_block(b, 0, half, half),
        extract_block(b, half, 0, half),
        extract_block(b, half, half, half),
    );

    let compute = |x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>| {
        strassen_implementation(x, y, depth + 1, max_parallel_depth)
    };

    let (p1, p2, p3, p4, p5, p6, p7) =
        if depth < max_parallel_depth {
            let (p1, p2) = rayon::join(
                || compute(&add_blocks(&a11, &a22), &add_blocks(&b11, &b22)),
                || compute(&add_blocks(&a21, &a22), &b11),
            );

            let (p3, p4) = rayon::join(
                || compute(&a11, &sub_blocks(&b12, &b22)),
                || compute(&a22, &sub_blocks(&b21, &b11)),
            );

            let (p5, p6) = rayon::join(
                || compute(&add_blocks(&a11, &a12), &b22),
                || compute(&sub_blocks(&a21, &a11), &add_blocks(&b11, &b12)),
            );

            let p7 = compute(
                &sub_blocks(&a12, &a22),
                &add_blocks(&b21, &b22),
            );

            (p1, p2, p3, p4, p5, p6, p7)
        } else {
            (
                compute(&add_blocks(&a11, &a22), &add_blocks(&b11, &b22)),
                compute(&add_blocks(&a21, &a22), &b11),
                compute(&a11, &sub_blocks(&b12, &b22)),
                compute(&a22, &sub_blocks(&b21, &b11)),
                compute(&add_blocks(&a11, &a12), &b22),
                compute(&sub_blocks(&a21, &a11), &add_blocks(&b11, &b12)),
                compute(&sub_blocks(&a12, &a22), &add_blocks(&b21, &b22)),
            )
        };

    let c11 = add_blocks(
        &sub_blocks(&add_blocks(&p1, &p4), &p5),
        &p7,
    );

    let c12 = add_blocks(&p3, &p5);
    let c21 = add_blocks(&p2, &p4);

    let c22 = add_blocks(
        &add_blocks(&sub_blocks(&p1, &p2), &p3),
        &p6,
    );

    insert_block(&mut result, &c11, 0, 0);
    insert_block(&mut result, &c12, 0, half);
    insert_block(&mut result, &c21, half, 0);
    insert_block(&mut result, &c22, half, half);

    result
}


fn create_random_matrix(
    rows: usize, 
    cols: usize
) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();

    let mut matrix = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            matrix[i][j] = rng.gen_range(0.0..100.0);
        }
    }

    matrix
}

fn compare_matrices(
    a: &Vec<Vec<f64>>, 
    b: &Vec<Vec<f64>>
) -> bool {
    let rows: usize = a.len();
    let cols = b.len();
    
    if rows != cols {
        return false;
    }
    
    const EPSILON: f64 = 1e-6;
    
    for i in 0..rows {
        for j in 0..rows {
            if (a[i][j] - b[i][j]).abs() > EPSILON {
                return false;
            }
        }
    }
    
    true
}


fn extract_block(
    matrix: &Vec<Vec<f64>>,
    row_offset: usize,
    col_offset: usize,
    block_size: usize,
) -> Vec<Vec<f64>> {
    let mut block = vec![vec![0.0; block_size]; block_size];

    for r in 0..block_size {
        let src_row = row_offset + r;
        for c in 0..block_size {
            block[r][c] = matrix[src_row][col_offset + c];
        }
    }

    block
}


fn insert_block(
    target: &mut Vec<Vec<f64>>,
    block: &Vec<Vec<f64>>,
    row_offset: usize,
    col_offset: usize,
) {
    let block_size = block.len();

    for r in 0..block_size {
        let dst_row = row_offset + r;
        for c in 0..block_size {
            target[dst_row][col_offset + c] = block[r][c];
        }
    }
}


fn add_blocks(
    x: &Vec<Vec<f64>>, 
    y: &Vec<Vec<f64>>
) -> Vec<Vec<f64>> {
    let size = x.len();
    let mut sum = vec![vec![0.0; size]; size];

    for r in 0..size {
        for c in 0..size {
            sum[r][c] = x[r][c] + y[r][c];
        }
    }

    sum
}


fn sub_blocks(
    x: &Vec<Vec<f64>>, 
    y: &Vec<Vec<f64>>
) -> Vec<Vec<f64>> {
    let size = x.len();
    let mut diff = vec![vec![0.0; size]; size];

    for r in 0..size {
        for c in 0..size {
            diff[r][c] = x[r][c] - y[r][c];
        }
    }

    diff
}
