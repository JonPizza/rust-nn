use rand::Rng;

mod math;

const NN_SHAPE: [usize; 3] = [2, 8, 2];

const XOR_DATA: [[f32; 4]; 4] = [
    [1.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

struct NeuralNet {
    neuron_values: [Vec<f32>; 3],
    weight_mtrx_1: Vec<Vec<f32>>,
    weight_mtrx_2: Vec<Vec<f32>>,
    biases: [Vec<f32>; 2],
    learning_rate: f32,
    gradients: NNGradients,
}

struct NNGradients {
    wm1: Vec<Vec<f32>>,
    wm2: Vec<Vec<f32>>,
    bias: [Vec<f32>; 2],
}

fn create_nn() -> NeuralNet {
    let mut nn = NeuralNet {
        neuron_values: [vec![0.0; NN_SHAPE[0]], vec![0.0; NN_SHAPE[1]], vec![0.0; NN_SHAPE[2]]],
        weight_mtrx_1: Vec::<Vec<f32>>::new(),
        weight_mtrx_2: Vec::<Vec<f32>>::new(),
        biases: [vec![0.0; NN_SHAPE[1]], vec![0.0; NN_SHAPE[2]]],
        learning_rate: 0.0,
        gradients: NNGradients {
            wm1: Vec::<Vec<f32>>::new(),
            wm2: Vec::<Vec<f32>>::new(),
            bias: [vec![0.0; NN_SHAPE[1]], vec![0.0; NN_SHAPE[2]]],
        }
    };
    init(&mut nn);
    nn
}

fn init(nn: &mut NeuralNet) {
    let mut rng = rand::thread_rng();

    nn.weight_mtrx_1.clear();
    nn.gradients.wm1.clear();
    for i in 0..NN_SHAPE[0] {
        nn.weight_mtrx_1.push(Vec::<f32>::new());
        nn.gradients.wm1.push(Vec::<f32>::new());
        for _ in 0..NN_SHAPE[1] {
            nn.weight_mtrx_1[i].push(rng.gen::<f32>());
            nn.gradients.wm1[i].push(0.0);
        }
    }

    nn.weight_mtrx_2.clear();
    nn.gradients.wm2.clear();
    for i in 0..NN_SHAPE[1] {
        nn.weight_mtrx_2.push(Vec::<f32>::new());
        nn.gradients.wm2.push(Vec::<f32>::new());
        for _ in 0..NN_SHAPE[2] {
            nn.weight_mtrx_2[i].push(rng.gen::<f32>());
            nn.gradients.wm2[i].push(0.0);
        }
    }

    nn.gradients.bias = [vec![0.0; NN_SHAPE[1]], vec![0.0; NN_SHAPE[2]]];
    for bias_vec in nn.biases.iter_mut() {
        for n in bias_vec.iter_mut() {
            *n = rng.gen::<f32>();
        }
    }

    nn.learning_rate = 0.1;
}

fn feed_forward(nn: &mut NeuralNet, input_vec: Vec<f32>) -> Vec<f32> {
    nn.neuron_values[0] = input_vec.clone();
    let mut state: Vec<f32> = math::mmul(input_vec, (*nn.weight_mtrx_1).to_vec());
    state = math::sum_vec(&mut state, &nn.biases[0]).to_vec();
    state = math::sigmoid_vec(state);
    nn.neuron_values[1] = state.clone();
    state = math::mmul(state, (*nn.weight_mtrx_2).to_vec());
    state = math::sum_vec(&mut state, &nn.biases[1]).to_vec();
    state = math::sigmoid_vec(state);
    nn.neuron_values[2] = state.clone();
    state
}

fn cost(actual_out: Vec<f32>, expected_out: &Vec<f32>) -> f32 {
    assert!(actual_out.len() == expected_out.len());
    let mut c: f32 = 0.0;
    for i in 0..actual_out.len() {
        c += (expected_out[i] - actual_out[i]).powf(2.0) / 2.0;
    }
    c / (actual_out.len() as f32)
}

fn backprop(nn: &mut NeuralNet, expected_out: Vec<f32>) {
    for i in 0..NN_SHAPE[2] {
        for j in 0..NN_SHAPE[1] {
            let first_partial: f32 = (expected_out[i] - nn.neuron_values[2][i]) 
                                        * math::sigmoid_prime(nn.neuron_values[1][j] * nn.weight_mtrx_2[j][i] + nn.biases[1][i]);
            nn.gradients.wm2[j][i] += first_partial * nn.neuron_values[1][j];
            nn.gradients.bias[1][i] += first_partial;
            for k in 0..NN_SHAPE[0] {
                nn.gradients.wm1[k][j] += first_partial * nn.weight_mtrx_2[j][i] 
                    * math::sigmoid_prime(nn.neuron_values[0][k] * nn.weight_mtrx_1[k][j] + nn.biases[0][j]) * nn.neuron_values[0][k];
                nn.gradients.bias[0][j] += first_partial * nn.weight_mtrx_2[j][i] 
                    * math::sigmoid_prime(nn.neuron_values[0][k] * nn.weight_mtrx_1[k][j] + nn.biases[0][j]);
            }
        }
    }
}

fn clear_gradient(g: &mut NNGradients) {
    for i in g.wm1.iter_mut() {
        for w in i.iter_mut() {
            *w = 0.0;
        }
    }

    for i in g.wm2.iter_mut() {
        for w in i.iter_mut() {
            *w = 0.0;
        }
    }

    g.bias = [vec![0.0; NN_SHAPE[1]], vec![0.0; NN_SHAPE[2]]];
}

fn apply_gradient(nn: &mut NeuralNet) {
    for i in 0..nn.weight_mtrx_1.len() {
        for j in 0..nn.weight_mtrx_1[i].len() {
            nn.weight_mtrx_1[i][j] += nn.gradients.wm1[i][j] * nn.learning_rate;
        }
    }

    for i in 0..nn.weight_mtrx_2.len() {
        for j in 0..nn.weight_mtrx_2[i].len() {
            nn.weight_mtrx_2[i][j] += nn.gradients.wm2[i][j] * nn.learning_rate;
        }
    }

    for i in 0..nn.biases.len() {
        for j in 0..nn.biases[i].len() {
            nn.biases[i][j] += nn.gradients.bias[i][j] * nn.learning_rate;
        }
    }
    clear_gradient(&mut nn.gradients);
}

fn train(nn: &mut NeuralNet, iterations: i32) {
    let mut cost_avg: f32 = 0.0;
    for i in 0..iterations {
        for j in 0..XOR_DATA.len() {
            feed_forward(nn, vec![XOR_DATA[j][0], XOR_DATA[j][1]]);
            backprop(nn, vec![XOR_DATA[j][2], XOR_DATA[j][3]]);
            println!("{:?} --> {:?}", vec![XOR_DATA[j][0], XOR_DATA[j][1]], nn.neuron_values[2]);
            cost_avg += cost(vec![XOR_DATA[j][2], XOR_DATA[j][3]], &nn.neuron_values[2]);
        }
        apply_gradient(nn);    
        println!("Cost on iteration {} --> {}", i, cost_avg / 4.0);
        cost_avg = 0.0;
    }
}

pub fn testie() {
    let mut nn = create_nn();
    init(&mut nn);
    let input_vec: Vec<f32> = vec![1.0, 1.0];
    train(&mut nn, 100000000);
}