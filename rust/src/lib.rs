use numpy::{ IntoPyArray, PyArray2, PyArrayLike2, PyArrayMethods };
use pyo3::prelude::*;
use numpy::ndarray::{ Array2 };
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
#[pyfunction]
fn get_stat_protocol<'py>(
    py: Python<'py>,
    array: PyArrayLike2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.to_owned_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
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
    Ok(output.into_pyarray(py).into())
}

#[pymodule(name = "numquant")]
fn rolling_funcs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_stat_protocol, m)?)?;
    Ok(())
}
