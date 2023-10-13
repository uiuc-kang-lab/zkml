#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use group::ff::Field;
use criterion::{BenchmarkId, Criterion};

use halo2_proofs::{
  halo2curves::bn256::Fr,
  arithmetic::best_fft,
  poly::EvaluationDomain
};
use rand_core::OsRng;

pub fn bench_kzg_fft(c: &mut Criterion) {
  let j = 5;
  let mut group = c.benchmark_group("kzg_fft");
  for k in 13..20 {
    let domain = EvaluationDomain::new(j,k);
    let omega = domain.get_omega();
    let l = 1<<k;
    let data = domain.get_fft_data(l);
    group.bench_function(BenchmarkId::new("k", k), |b| {
        let mut a = (0..(1 << k)).map(|_| Fr::random(OsRng)).collect::<Vec<_>>();
        b.iter(|| {
            best_fft(&mut a, omega, k as u32, data, false);
        });
    });
  }
}

criterion_group!{
  name=benches; 
  config=Criterion::default().sample_size(10); 
  targets=bench_kzg_fft
}
criterion_main!(benches);