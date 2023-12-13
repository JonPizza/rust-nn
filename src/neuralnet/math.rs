pub fn mmul(a: Vec<f32>, b: Vec<Vec<f32>>) -> Vec<f32> {
    // a -> nx1^T, b -> nxm
    assert!(a.len() == b.len());
    let mut output_vec: Vec<f32> = Vec::<f32>::new();
    for i in 0..b[0].len() {
        output_vec.push(0.0);
        for j in 0..a.len() {
            output_vec[i] += b[j][i] * a[j];
        }
    }
    output_vec
}

pub fn sigmoid(x: f32) -> f32 {
    let e: f32 = 2.71;
    1.0 / (1.0 + e.powf(-1.0 * x))
}

pub fn sigmoid_prime(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn sigmoid_vec(x: Vec<f32>) -> Vec<f32> {
    let mut out_vec: Vec<f32> = Vec::<f32>::new();
    for val in x {
        out_vec.push(sigmoid(val));
    }
    out_vec
}

pub fn sum_vec<'a, 'b>(a: &'a mut Vec<f32>, b: &'b Vec<f32>) -> &'a Vec<f32> {
    assert!(a.len() == b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
    a
}