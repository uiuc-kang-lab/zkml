//! This is a module for dispatching between different FFT implementations at runtime based on environment variable `FFT`.

use std::env::var;

use ff::Field;

use self::recursive::FFTData;
use crate::{arithmetic::FftGroup, plonk::log_info};

pub mod baseline;
pub mod parallel;
pub mod recursive;

/// Runtime dispatcher to concrete FFT implementation
pub fn fft<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    match var("FFT") {
        Err(_) => {
            // No `FFT=` environment variable specified.
            log_info("=== Parallel FFT ===".to_string());
            parallel::fft(a, omega, log_n, data, inverse)
        }
        Ok(fft_impl) if fft_impl == "baseline"=> {
            log_info("=== Baseline FFT ===".to_string());
            baseline::fft(a, omega, log_n, data, inverse)
        }
        Ok(fft_impl) if fft_impl == "recursive" => {
            log_info("=== Recusive FFT ===".to_string());
            recursive::fft(a, omega, log_n, data, inverse)
        }
        Ok(fft_impl) if fft_impl == "parallel" => {
            log_info("=== Parallel FFT ===".to_string());
            parallel::fft(a, omega, log_n, data, inverse)
        }
        _ => {
            panic!("Please either specify environment variable `FFT={{baseline,recursive,parallel}}` or remove it all together.")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{time::Instant, env::var};

    use ff::Field;
    use halo2curves::bn256::Fr as Scalar;
    use rand_core::OsRng;

    use crate::{
        fft::{self, recursive::FFTData},
        multicore,
        arithmetic::{eval_polynomial, lagrange_interpolate, best_fft},
        plonk::{start_measure, log_info, stop_measure},
        poly::EvaluationDomain,
    };

    /// Read Environment Variable `DEGREE`
    fn get_degree() -> usize {
        var("DEGREE")
            .unwrap_or_else(|_| "22".to_string())
            .parse()
            .expect("Cannot parse DEGREE env var as usize")
    }


    #[test]
    fn test_fft_parallel() {
        let max_log_n = 22;
        let min_log_n = 8;
        let a = (0..(1 << max_log_n))
            .into_iter()
            .map(|i| Scalar::from(i as u64))
            .collect::<Vec<_>>();

        log_info("\n---------- test_fft_parallel ---------".to_owned());
        for log_n in min_log_n..=max_log_n {
            let domain = EvaluationDomain::<Scalar>::new(1, log_n);
            let mut a0 = a[0..(1 << log_n)].to_vec();
            let mut a1 = a0.clone();

            // FFTData is not used in `baseline` and `parallel` so default values suffices.
            let d = FFTData::default();
            let f = false;

            // warm up & correct test
            fft::baseline::fft(&mut a0, domain.get_omega(), log_n, &d, f);
            fft::parallel::fft(&mut a1, domain.get_omega(), log_n, &d, f);
            assert_eq!(a0, a1);

            let ori_time = Instant::now();
            fft::baseline::fft(&mut a0, domain.get_omega(), log_n, &d, f);
            let ori_time = ori_time.elapsed();
            let ori_micros = f64::from(ori_time.as_micros() as u32);

            let opt_time = Instant::now();
            fft::parallel::fft(&mut a1, domain.get_omega(), log_n, &d, f);
            let opt_time = opt_time.elapsed();
            let opt_micros = f64::from(opt_time.as_micros() as u32);

            log_info(format!(
                "    [log_n = {}] orig::fft time: {:?}, scroll::fft time: {:?}, speedup: {}",
                log_n,
                ori_time,
                opt_time,
                ori_micros / opt_micros
            ));
        }
    }

    #[test]
    fn test_fft_recursive() {
        log_info("\n---------- test_fft_recursive ---------".to_owned());

        let k = get_degree() as u32;

        let domain = EvaluationDomain::<Scalar>::new(1, k);
        let n = domain.get_n() as usize;

        let input = vec![Scalar::random(OsRng); n];

        let num_threads = multicore::current_num_threads();

        let mut a = input.clone();
        let l_a= a.len();
        let start = start_measure(format!("best fft {} ({})", a.len(), num_threads), false);
        fft::baseline::fft(&mut a, domain.get_omega(), k, domain.get_fft_data(l_a), false);
        stop_measure(start);

        let mut b = input;
        let l_b= b.len();
        let start = start_measure(
            format!("recursive fft {} ({})", a.len(), num_threads),
            false,
        );
        fft::recursive::fft(&mut b, domain.get_omega(), k, domain.get_fft_data(l_b), false);
        stop_measure(start);

        for i in 0..n {
            //log_info(format!("{}: {} {}", i, a[i], b[i]));
            assert_eq!(a[i], b[i]);
        }
    }

    #[test]
    fn test_fft_all() {
        log_info("\n---------- test_fft_all ---------".to_owned());

        let k = get_degree() as u32;

        let domain = EvaluationDomain::<Scalar>::new(1, k);
        let n = domain.get_n() as usize;

        let input = vec![Scalar::random(OsRng); n];

        let num_threads = multicore::current_num_threads();

        let mut data_baseline = input.clone();
        let l_baseline = data_baseline.len();
        let start = start_measure(
            format!("baseline  fft {} ({})", data_baseline.len(), num_threads),
            false,
        );
        fft::baseline::fft(&mut data_baseline, domain.get_omega(), k, domain.get_fft_data(l_baseline), false);
        stop_measure(start);

        let mut data_parallel = input.clone();
        let l_parallel = data_parallel.len();
        let start = start_measure(
            format!("parallel  fft {} ({})", data_parallel.len(), num_threads),
            false,
        );
        fft::parallel::fft(&mut data_parallel, domain.get_omega(), k, domain.get_fft_data(l_parallel), false);
        stop_measure(start);

        let mut data_recursive = input;
        let l_recursive = data_recursive.len();
        let start = start_measure(
            format!("recursive fft {} ({})", data_recursive.len(), num_threads),
            false,
        );
        fft::recursive::fft(
            &mut data_recursive,
            domain.get_omega(),
            k,
            domain.get_fft_data(l_recursive),
            false,
        );
        stop_measure(start);

        for i in 0..n {
            // log_info(format!("{}: {} {}", i, data_baseline[i], data_recursive[i]));
            assert_eq!(data_baseline[i], data_recursive[i]);
            // log_info(format!("{}: {} {}", i, data_baseline[i], data_parallel[i]));
            assert_eq!(data_baseline[i], data_parallel[i]);
        }
    }

    #[test]
    fn test_fft_single() {
        log_info("\n---------- test_fft_single ---------".to_owned());

        let k = get_degree() as u32;

        let domain = EvaluationDomain::new(1, k);
        let n = domain.get_n() as usize;

        let mut input = vec![Scalar::random(OsRng); n];
        let l = input.len();

        let num_threads = multicore::current_num_threads();

        let start = start_measure(format!("fft {} ({})", input.len(), num_threads), false);
        fft::fft(&mut input, domain.get_omega(), k, domain.get_fft_data(l), false);
        stop_measure(start);
    }

    #[test]
    fn test_mem_leak() {
        let j = 1;
        let k = 3;
        let domain = EvaluationDomain::new(j,k);
        let omega = domain.get_omega();
        let l = 1<<k;
        let data = domain.get_fft_data(l);
        let mut a = (0..(1 << k)).map(|_| Scalar::random(OsRng)).collect::<Vec<_>>();

        best_fft(&mut a, omega, k as u32, data, false);
}

}
