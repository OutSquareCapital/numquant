use ndarray::Array2;

struct AccumulatorBase<T> {
    multiplier: T,
    sum: T,
    compensation: T,
}

impl AccumulatorBase<f32> {
    fn new(multiplier: f32) -> Self {
        AccumulatorBase {
            multiplier: multiplier,
            sum: 0.0,
            compensation: 0.0,
        }
    }

    fn add_contribution(&mut self, value: f32) {
        let temp: f32 = value.powf(self.multiplier) - self.compensation;
        let total: f32 = self.sum + temp;
        self.compensation = total - self.sum - temp;
        self.sum = total;
    }

    fn remove_contribution(&mut self, value: f32) {
        let temp: f32 = -value.powf(self.multiplier) - self.compensation;
        let total: f32 = self.sum + temp;
        self.compensation = total - self.sum - temp;
        self.sum = total;
    }
}

type AccumulatorFloat32 = AccumulatorBase<f32>;

fn get_stat_protocol(array: &Array2<f32>, length: usize, min_length: usize) -> Array2<f32> {
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::from_elem((num_rows, num_cols), f32::NAN);
    for col in 0..num_cols {
        let mut accumulator = AccumulatorFloat32::new(1.0);
        let mut observation_count: i32 = 0;
        for row in 0..num_rows {
            if !array[[row, col]].is_nan() {
                observation_count += 1;
                accumulator.add_contribution(array[[row, col]]);
            }
            if row > length {
                let idx: usize = row - length;
                if !array[[idx, col]].is_nan() {
                    observation_count -= 1;
                    accumulator.remove_contribution(array[[idx, col]]);
                }
            }
            if observation_count >= (min_length as i32) {
                output[[row, col]] = accumulator.sum / (observation_count as f32);
            }
        }
    }
    output
}

fn main() {
    let data = Array2::from_shape_vec(
        (30, 1),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 12.0, 8.0, 9.0, 10.0, 12.0, 12.0, 13.0, 14.0, 12.0, 1.0,
            2.0, 3.0, 4.0, 5.0, 6.0, 12.0, 8.0, 9.0, 10.0, 12.0, 12.0, 13.0, 14.0, 12.0,
        ],
    )
    .unwrap();

    let result = get_stat_protocol(&data, 5, 5);
    println!("{:?}", result);
}
